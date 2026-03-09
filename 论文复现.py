import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile


def _get_component_names(domain):
    names = getattr(domain, 'component_names', None)
    if names is not None:
        return list(names)
    n_components = getattr(domain, 'n_components', None)
    component_name = getattr(domain, 'component_name', None)
    if n_components is not None and component_name is not None:
        return [component_name(i) for i in range(int(n_components))]
    return []


def _get_profile(domain, name):
    values = getattr(domain, 'values', None)
    if values is not None:
        try:
            return np.asarray(values(name), dtype=float)
        except Exception:
            pass

    value = getattr(domain, 'value', None)
    grid = getattr(domain, 'grid', None)
    if value is None or grid is None:
        raise AttributeError(f"无法从域对象读取组分/温度剖面: {name}")

    component_index = getattr(domain, 'component_index', None)
    if component_index is not None:
        try:
            idx = component_index(name)
            return np.array([value(idx, i) for i in range(len(grid))], dtype=float)
        except Exception:
            pass

    return np.array([value(name, i) for i in range(len(grid))], dtype=float)


def _get_first_available_profile(domain, candidates):
    component_names = _get_component_names(domain)
    for name in candidates:
        if not component_names or name in component_names:
            try:
                return _get_profile(domain, name)
            except Exception:
                continue
    raise KeyError(f"找不到可用分量: {candidates}")


def _get_flow_domain(sim):
    for attr in ('flame', 'flow'):
        domain = getattr(sim, attr, None)
        if domain is not None and hasattr(domain, 'grid'):
            return domain

    for domain in getattr(sim, 'domains', []):
        if hasattr(domain, 'grid'):
            component_names = _get_component_names(domain)
            if not component_names or 'T' in component_names:
                return domain

    domains = getattr(sim, 'domains', None)
    if domains and len(domains) > 1:
        return domains[1]
    raise AttributeError('无法定位流动域（flow/flame domain）')


def _is_ascii_path(path):
    try:
        os.path.abspath(path).encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def _get_cantera_io_path(path):
    abs_path = os.path.abspath(path)
    if _is_ascii_path(abs_path):
        return abs_path

    cache_dir = os.path.join(tempfile.gettempdir(), 'cantera_1d_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, os.path.basename(abs_path))


def _restore_solution(sim, filename):
    filename = _get_cantera_io_path(filename)
    try:
        sim.restore(filename, name='solution')
    except TypeError:
        sim.restore(filename)


def _save_solution(sim, filename):
    filename = _get_cantera_io_path(filename)
    try:
        sim.save(filename, name='solution', overwrite=True)
    except TypeError:
        try:
            sim.save(filename, 'solution', overwrite=True)
        except TypeError:
            sim.save(filename, overwrite=True)

# ==========================================
# 1. 基础参数
# ==========================================
mechanism_file = 'gri30.yaml'
domain_width = 0.02
target_u_inlet = 10.0
ignite_u_inlet = 2
target_wall_temp = 300.0
Tin_ignite = 1200.0
Twall_ignite = 1200.0
k_levels = [0.1, 0.2, 0.3, 0.4]

# ==========================================
# 2. 气体生成 (保持不变)
# ==========================================
def get_diluted_gas(k):
    fuel, oxidizer = 'CH4', 'O2:1.0, N2:3.76'
    T_fresh, P = 298.0, ct.one_atm

    gas_fresh = ct.Solution(mechanism_file)
    gas_fresh.TP = T_fresh, P
    gas_fresh.set_equivalence_ratio(1.0, fuel, oxidizer)

    gas_prod = ct.Solution(mechanism_file)
    gas_prod.TP = T_fresh, P
    gas_prod.set_equivalence_ratio(1.0, fuel, oxidizer)
    gas_prod.equilibrate('HP')

    if k == 0: return gas_fresh
    if k == 1: return gas_prod

    Q_fresh = ct.Quantity(gas_fresh, mass=(1 - k), constant='HP')
    Q_prod = ct.Quantity(gas_prod, mass=k, constant='HP')
    Q_mix = Q_fresh + Q_prod

    gas_inlet = ct.Solution(mechanism_file)
    gas_inlet.TPY = Q_mix.T, Q_mix.P, Q_mix.Y
    return gas_inlet

# ==========================================
# 3. 模拟计算 (保持不变)
# ==========================================
def run_simulation(k):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"new_result_k{k}.yaml")
    cache_filename = _get_cantera_io_path(filename)
    if os.path.exists(cache_filename):
        print(f"[读取] 发现存档 {cache_filename}...")
        gas = ct.Solution(mechanism_file)
        sim = ct.ImpingingJet(gas=gas, width=domain_width)
        _restore_solution(sim, cache_filename)
        return sim

    print(f"\n[计算] 开始模拟 k={k} ...")
    gas_inlet = get_diluted_gas(k)
    sim = ct.ImpingingJet(gas=gas_inlet, width=domain_width)
    #第一阶段：低速高温点火过程
    gas_ignite = ct.Solution(mechanism_file)
    gas_ignite.TPY = float(max(Tin_ignite,gas_inlet.T)), ct.one_atm, gas_inlet.Y
    sim.inlet.mdot = gas_ignite.density * ignite_u_inlet
    sim.inlet.T = gas_ignite.T
    sim.inlet.Y = gas_ignite.Y
    sim.surface.T = Twall_ignite
    sim.set_refine_criteria(ratio=3.0, slope=0.05, curve=0.05, prune=0.0)
    sim.max_grid_points=10000
    sim.max_time_step_count=10000
    sim.set_grid_min(1e-9)
    sim.set_initial_guess(products="equil")
    print("低速高温求解点火过程...")
    sim.solve(loglevel=1, auto=True)
    #第二阶段：逐步恢复至目标温度
    Tin_path = np.linspace(float(max(gas_inlet.T, Tin_ignite)), float(gas_inlet.T), 5)
    Twall_path = np.linspace(float(Twall_ignite), float(target_wall_temp), 8)

    n = min(len(Tin_path), len(Twall_path))
    print("入口温度恢复中...")
    for i in range(1, n):
        sim.inlet.T = float(Tin_path[i])
        sim.surface.T = float(Twall_path[i])
        sim.solve(loglevel=1, refine_grid=True, auto=True)

    # finish wall ramp if needed
    print("壁面温度恢复中...")
    for j in range(n, len(Twall_path)):
        sim.surface.T = float(Twall_path[j])
        sim.solve(loglevel=1, refine_grid=True, auto=False)


    u_path = np.linspace(float(ignite_u_inlet), float(target_u_inlet), 6)
    gas_ramp = ct.Solution(mechanism_file)
    gas_ramp.TPY = float(gas_inlet.T),ct.one_atm, gas_inlet.Y
    print("入口速度恢复中...")
    for i in range(1, len(u_path)):
        sim.inlet.mdot = gas_ramp.density * float(u_path[i])
        sim.solve(loglevel=1, refine_grid=True, auto=False)
    _save_solution(sim, cache_filename)
    return sim

# ==========================================
# 4. 后处理模块 (核心修正：加入密度项)
# ==========================================
def calc_emissions(sim):
    flow = _get_flow_domain(sim)
    z = flow.grid

    # 1. 获取 spreadRate (V = v/r)
    # 它的单位是 1/s，必须乘以密度才能得到质量通量
    V = _get_first_available_profile(flow, ('spreadRate', 'V'))

    # 2. 获取温度 T (用于计算密度)
    T = _get_profile(flow, 'T')

    # 3. 获取组分 Y
    n_species = sim.gas.n_species
    n_points = len(z)
    Y = np.zeros((n_species, n_points))
    for k, name in enumerate(sim.gas.species_names):
        Y[k, :] = _get_profile(flow, name)

    # 4. 积分计算
    mw = sim.gas.molecular_weights
    i_co = sim.gas.species_index('CO')
    i_o2 = sim.gas.species_index('O2')
    i_h2o = sim.gas.species_index('H2O')

    m_co = 0.0; m_tot = 0.0; m_o2 = 0.0; m_h2o = 0.0

    # 临时气体对象，用于计算每一点的密度
    gas_calc = ct.Solution(mechanism_file)

    for i in range(len(z) - 1):
        dz = z[i+1] - z[i]

        # 取中间点的状态
        T_mid = 0.5 * (T[i] + T[i+1])
        Y_mid = 0.5 * (Y[:,i] + Y[:,i+1])
        V_mid = 0.5 * (V[i] + V[i+1])

        # 【关键修正】计算这一点的密度 rho
        gas_calc.TPY = T_mid, sim.gas.P, Y_mid
        rho_mid = gas_calc.density

        # 【关键修正】质量通量 = 密度 * (v/r) * dz
        # 注意：公式里的系数 2 在分子分母会约掉，所以不乘也没关系，但为了物理清晰可以乘上
        # 这里我们算的是加权权重，只要分子分母统一即可。
        mass_flux_out = rho_mid * V_mid * dz

        # 混合物分子量
        MW_mix = gas_calc.mean_molecular_weight # Cantera自带有这个属性，直接用更准

        # 转化为摩尔流率
        mole_flux_total = mass_flux_out / MW_mix
        m_tot += mole_flux_total

        # 累加各组分摩尔流率
        # mole_flux_k = mass_flux_out * Y_k / MW_k
        m_co  += mass_flux_out * Y_mid[i_co] / mw[i_co]
        m_o2  += mass_flux_out * Y_mid[i_o2] / mw[i_o2]
        m_h2o += mass_flux_out * Y_mid[i_h2o] / mw[i_h2o]

    if m_tot == 0: return 0.0

    X_co_wet = m_co / m_tot
    X_h2o_wet = m_h2o / m_tot
    X_o2_wet = m_o2 / m_tot

    dry_div = (1.0 - X_h2o_wet)
    if dry_div == 0: dry_div = 1.0
    X_co_dry = X_co_wet / dry_div
    X_o2_dry = X_o2_wet / dry_div

    if X_o2_dry >= 0.209: corr = 1.0
    else: corr = (0.209 - 0.15) / (0.209 - X_o2_dry)

    return X_co_dry * corr * 1e6

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    results_k = []
    results_co = []

    for k in k_levels:
        try:
            sim = run_simulation(k)
            co_ppm = calc_emissions(sim)
            results_k.append(k)
            results_co.append(co_ppm)
            print(f"   -> [结果] k={k}, CO={co_ppm:.2f} ppm")
        except Exception as e:
            print(f"   -> [错误] k={k}: {e}")

    if len(results_k) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(results_k, results_co, 'o-', color='firebrick', label='Corrected Physics')
        plt.title(f'CO Emissions (Density Weighted)', fontsize=14)
        plt.xlabel('Dilution Ratio k')
        plt.ylabel('CO [ppmvd @ 15% O2]')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('final_corrected.png', dpi=300)
        plt.show()