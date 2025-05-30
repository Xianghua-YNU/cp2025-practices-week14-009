import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现van der Pol方程
    # dx/dt = v
    # dv/dt = mu(1-x^2)v - omega^2*x
    return np.array([v, mu*(1-x**2)*v - omega**2*x])



def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # TODO: 实现RK4方法
    k1 = dt * ode_func(state, t, **kwargs)
    k2 = dt * ode_func(state + k1/2, t + dt/2, **kwargs)
    k3 = dt * ode_func(state + k2/2, t + dt/2, **kwargs)
    k4 = dt * ode_func(state + k3, t + dt, **kwargs)
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state


def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, 
                   t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T
                  
def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='Position x(t)')
    plt.plot(t, states[:, 1], label='Velocity v(t)')
    plt.xlabel('Time t')
    plt.ylabel('State Variables')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5)
    plt.scatter(states[0, 0], states[0, 1], c='r', marker='o', s=100, label='起点')
    plt.scatter(states[-1, 0], states[-1, 1], c='g', marker='s', s=100, label='终点')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率
    
    返回:
        float: 系统的能量
    """
    # TODO: 实现能量计算
    # E = (1/2)v^2 + (1/2)omega^2*x^2
    x, v = state
    return 0.5 * v**2 + 0.5 * omega**2 * x**2

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # TODO: 实现极限环分析
    # 取最后1/3的数据进行分析（稳态部分）
    n = len(states)
    start_idx = n * 2 // 3
    x_vals = states[start_idx:, 0]
    
    # 计算振幅（取最大值作为振幅）
    amplitude = np.max(np.abs(x_vals))
    
    # 计算周期：通过寻找过零点
    zero_crossings = np.where(np.diff(np.sign(x_vals)))[0]
    if len(zero_crossings) < 2:
        return amplitude, np.nan
    
    # 计算相邻过零点之间的时间间隔（取平均作为周期）
    periods = np.diff(zero_crossings)
    period = np.mean(periods) * 0.01 * 2  # 乘以时间步长和2（因为相邻过零是半个周期）
    
    return amplitude, period


def plot_energy_evolution(t: np.ndarray, states: np.ndarray, omega: float, title: str) -> None:
    """
    绘制能量随时间的变化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        omega: float, 角频率
        title: str, 图标题
    """
    energy = np.array([calculate_energy(state, omega) for state in states])
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, energy, 'm-', linewidth=1.5)
    plt.xlabel('时间 t')
    plt.ylabel('能量 E')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 设置基本参数
    mu = 1.0
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # TODO: 任务1 - 基本实现
    # 1. 求解van der Pol方程
    # 2. 绘制时间演化图
    print("任务1：基本实现 (μ=1)")
    mu = 1.0
    t_arr, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    
    # 绘制时间演化图
    plot_time_evolution(t_arr, states, f"van der Pol振子时间演化 (μ={mu})")
    
    # 绘制相空间轨迹
    plot_phase_space(states, f"van der Pol振子相空间轨迹 (μ={mu})")
    
    # 计算并显示能量
    plot_energy_evolution(t_arr, states, omega, f"van der Pol振子能量演化 (μ={mu})")
    
    # 分析极限环
    amplitude, period = analyze_limit_cycle(states)
    print(f"μ={mu}: 振幅={amplitude:.4f}, 周期={period:.4f}")
    
    # TODO: 任务2 - 参数影响分析
    # 1. 尝试不同的mu值
    # 2. 比较和分析结果

    print("\n任务2：参数影响分析")
    mu_values = [1.0, 2.0, 4.0]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(12, 8))
    for i, mu in enumerate(mu_values):
        # 求解ODE
        t_arr, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        
        # 绘制时间演化
        plt.subplot(2, 1, 1)
        plt.plot(t_arr, states[:, 0], color=colors[i], label=f'μ={mu}')
        
        # 绘制相空间轨迹
        plt.subplot(2, 1, 2)
        plt.plot(states[:, 0], states[:, 1], color=colors[i], label=f'μ={mu}')
        
        # 分析极限环特性
        amplitude, period = analyze_limit_cycle(states)
        print(f"μ={mu}: 振幅={amplitude:.4f}, 周期={period:.4f}")
    
    # 设置子图1（时间演化）
    plt.subplot(2, 1, 1)
    plt.xlabel('时间 t')
    plt.ylabel('位置 x')
    plt.title('不同μ值的van der Pol振子时间演化')
    plt.grid(True)
    plt.legend()
    
    # 设置子图2（相空间轨迹）
    plt.subplot(2, 1, 2)
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title('不同μ值的van der Pol振子相空间轨迹')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # TODO: 任务3 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 分析极限环特征

    print("\n任务3：初始条件影响分析 (μ=4)")
    mu = 4.0
    initial_conditions = [
        np.array([1.0, 0.0]),   # 基准条件
        np.array([0.1, 0.0]),   # 小振幅
        np.array([2.0, 0.0]),   # 大振幅
        np.array([0.0, 1.5])    # 不同相位
    ]
    
    plt.figure(figsize=(10, 8))
    for i, ic in enumerate(initial_conditions):
        t_arr, states = solve_ode(van_der_pol_ode, ic, t_span, dt, mu=mu, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'初始条件: ({ic[0]}, {ic[1]})')
    
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(f'不同初始条件下的相空间轨迹 (μ={mu})')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    
    # TODO: 任务4 - 能量分析
    # 1. 计算和绘制能量随时间的变化
    # 2. 分析能量的耗散和补充

    print("\n任务4：能量分析 (μ=2)")
    mu = 2.0
    t_arr, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_energy_evolution(t_arr, states, omega, f"van der Pol振子能量演化 (μ={mu})")


if __name__ == "__main__":
    main()
