#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 学生代码模板

学生姓名：[余维]
学号：[20231050005]
完成日期：[2025/5/28]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lotka_volterra_system(state: np.ndarray, t: float, alpha: float, beta: float, 
                          gamma: float, delta: float) -> np.ndarray:
    """
    Lotka-Volterra方程组的右端函数
    
    方程组：
    dx/dt = α*x - β*x*y  (猎物增长率 - 被捕食率)
    dy/dt = γ*x*y - δ*y  (捕食者增长率 - 死亡率)
    
    参数:
        state: np.ndarray, 形状为(2,), 当前状态向量 [x, y]
        t: float, 时间（本系统中未显式使用，但保持接口一致性）
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
    
    返回:
        np.ndarray, 形状为(2,), 导数向量 [dx/dt, dy/dt]
    """
    x, y = state
    
    # TODO: 实现Lotka-Volterra方程组 (约2-3行代码)
    # 提示：根据上面的方程组计算 dx/dt 和 dy/dt
    # [STUDENT_CODE_HERE]
    x, y = state
    # 修正dxdt的计算，确保符号正确
    dxdt = alpha * x - beta * x * y  # 当x=y=1, alpha=1, beta=0.5时，dxdt = 1 - 0.5 = 0.5
    dydt = gamma * x * y - delta * y  # 当x=y=1, gamma=0.5, delta=2时，dydt = 0.5 - 2 = -1.5
    
    return np.array([dxdt, dydt])


def euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                 dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    欧拉法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数，签名为 f(y, t, *args)
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现欧拉法迭代 (约3-5行代码)
    # 提示：y_{n+1} = y_n + dt * f(y_n, t_n)
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        y[i+1] = y[i] + dt * f(y[i], t[i], *args)
    
    return t, y


def improved_euler_method(f, y0: np.ndarray, t_span: Tuple[float, float], 
                         dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    改进欧拉法（2阶Runge-Kutta法）求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现改进欧拉法 (约6-8行代码)
    # 提示：
    # k1 = h * f(y_n, t_n)
    # k2 = h * f(y_n + k1, t_n + h)
    # y_{n+1} = y_n + (k1 + k2) / 2
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1, ti + h, *args)
        
        y[i+1] = yi + (k1 + k2) / 2
    
    return t, y


def runge_kutta_4(f, y0: np.ndarray, t_span: Tuple[float, float], 
                  dt: float, *args) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶龙格-库塔法求解常微分方程组
    
    参数:
        f: 微分方程组的右端函数
        y0: np.ndarray, 初始条件向量
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        *args: 传递给f的额外参数
    
    返回:
        t: np.ndarray, 时间数组
        y: np.ndarray, 解数组，形状为 (len(t), len(y0))
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    n_vars = len(y0)
    
    y = np.zeros((n_steps, n_vars))
    y[0] = y0
    
    # TODO: 实现4阶龙格-库塔法 (约8-12行代码)
    # 提示：
    # k1 = h * f(y_n, t_n)
    # k2 = h * f(y_n + k1/2, t_n + h/2)
    # k3 = h * f(y_n + k2/2, t_n + h/2)
    # k4 = h * f(y_n + k3, t_n + h)
    # y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    # [STUDENT_CODE_HERE]
    for i in range(n_steps - 1):
        h = dt
        yi = y[i]
        ti = t[i]
        
        k1 = h * f(yi, ti, *args)
        k2 = h * f(yi + k1/2, ti + h/2, *args)
        k3 = h * f(yi + k2/2, ti + h/2, *args)
        k4 = h * f(yi + k3, ti + h, *args)
        
        y[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y


def solve_lotka_volterra(alpha: float, beta: float, gamma: float, delta: float,
                        x0: float, y0: float, t_span: Tuple[float, float], 
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用4阶龙格-库塔法求解Lotka-Volterra方程组
    
    参数:
        alpha: float, 猎物自然增长率
        beta: float, 捕食效率
        gamma: float, 捕食者从猎物获得的增长效率
        delta: float, 捕食者自然死亡率
        x0: float, 初始猎物数量
        y0: float, 初始捕食者数量
        t_span: Tuple[float, float], 时间范围
        dt: float, 时间步长
    
    返回:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量数组
        y: np.ndarray, 捕食者种群数量数组
    """
    # TODO: 调用runge_kutta_4函数求解方程组 (约3-5行代码)
    # 提示：
    # 1. 构造初始条件向量 y0_vec = [x0, y0]
    # 2. 调用runge_kutta_4函数
    # 3. 从解中提取x和y分量
    # [STUDENT_CODE_HERE]
    y0_vec = np.array([x0, y0])
    t, solution = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, 
                               alpha, beta, gamma, delta)
    
    x = solution[:, 0]
    y = solution[:, 1]
    
    return t, x, y


def compare_methods(alpha: float, beta: float, gamma: float, delta: float,
                   x0: float, y0: float, t_span: Tuple[float, float], 
                   dt: float) -> dict:
    """
    比较三种数值方法求解Lotka-Volterra方程组
    
    参数:
        alpha, beta, gamma, delta: 模型参数
        x0, y0: 初始条件
        t_span: 时间范围
        dt: 时间步长
    
    返回:
        dict: 包含三种方法结果的字典，格式为：
        {
            'euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'improved_euler': {'t': t_array, 'x': x_array, 'y': y_array},
            'rk4': {'t': t_array, 'x': x_array, 'y': y_array}
        }
    """
    # TODO: 使用三种方法求解并返回结果字典 (约10-15行代码)
    # 提示：
    # 1. 构造初始条件向量和参数
    # 2. 分别调用三种方法
    # 3. 构造并返回结果字典
    # [STUDENT_CODE_HERE]
    y0_vec = np.array([x0, y0])
    args = (alpha, beta, gamma, delta)
    
    # 欧拉法
    t_euler, sol_euler = euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    
    # 改进欧拉法
    t_ie, sol_ie = improved_euler_method(lotka_volterra_system, y0_vec, t_span, dt, *args)
    
    # 4阶龙格-库塔法
    t_rk4, sol_rk4 = runge_kutta_4(lotka_volterra_system, y0_vec, t_span, dt, *args)
    
    return {
        'euler': {'t': t_euler, 'x': sol_euler[:, 0], 'y': sol_euler[:, 1]},
        'improved_euler': {'t': t_ie, 'x': sol_ie[:, 0], 'y': sol_ie[:, 1]},
        'rk4': {'t': t_rk4, 'x': sol_rk4[:, 0], 'y': sol_rk4[:, 1]}
    }


def plot_population_dynamics(t: np.ndarray, x: np.ndarray, y: np.ndarray, 
                           title: str = "Lotka-Volterra种群动力学") -> None:
    """
    绘制种群动力学图
    
    参数:
        t: np.ndarray, 时间数组
        x: np.ndarray, 猎物种群数量
        y: np.ndarray, 捕食者种群数量
        title: str, 图标题
    """
    # TODO: 绘制两个子图 (约15-20行代码)
    # 子图1：时间序列图（x和y随时间变化）
    # 子图2：相空间轨迹图（y vs x）
    # 提示：使用plt.subplot(1, 2, 1)和plt.subplot(1, 2, 2)
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(12, 5))
    
    # 时间序列图
    plt.subplot(1, 2, 1)
    plt.plot(t, x, 'b-', label='Prey (x)', linewidth=2)
    plt.plot(t, y, 'r-', label='Predator (y)', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Population')
    plt.title('Population vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 相空间轨迹图
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'g-', linewidth=2)
    plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
    plt.xlabel('Prey Population (x)')
    plt.ylabel('Predator Population (y)')
    plt.title('Phase Space Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_method_comparison(results: dict) -> None:
    """
    绘制不同数值方法的比较图
    
    参数:
        results: dict, compare_methods函数的返回结果
    """
    # TODO: 绘制方法比较图 (约20-30行代码)
    # 提示：
    # 1. 创建2x3的子图布局
    # 2. 上排：三种方法的时间序列图
    # 3. 下排：三种方法的相空间图
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(15, 10))
    
    methods = ['euler', 'improved_euler', 'rk4']
    method_names = ['Euler Method', 'Improved Euler', '4th-order Runge-Kutta']
    colors = ['blue', 'orange', 'green']
    
    # 时间序列比较
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(2, 3, i+1)
        t = results[method]['t']
        x = results[method]['x']
        y = results[method]['y']
        
        plt.plot(t, x, color=color, linestyle='-', label='Prey', linewidth=2)
        plt.plot(t, y, color=color, linestyle='--', label='Predator', linewidth=2)
        plt.xlabel('Time t')
        plt.ylabel('Population')
        plt.title(f'{name} - Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 相空间比较
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        plt.subplot(2, 3, i+4)
        x = results[method]['x']
        y = results[method]['y']
        
        plt.plot(x, y, color=color, linewidth=2)
        plt.plot(x[0], y[0], 'o', color=color, markersize=6)
        plt.xlabel('Prey Population (x)')
        plt.ylabel('Predator Population (y)')
        plt.title(f'{name} - Phase Space')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_parameters() -> None:
    """
    分析不同参数对系统行为的影响
    
    分析内容：
    1. 不同初始条件的影响
    2. 守恒量验证
    """
    # TODO: 实现参数分析 (约30-40行代码)
    # 提示：
    # 1. 设置基本参数
    # 2. 测试不同初始条件
    # 3. 计算并验证守恒量
    # 4. 绘制分析结果
    # [STUDENT_CODE_HERE]
    # 基本参数
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    plt.figure(figsize=(15, 10))
    
    # 不同初始条件的影响
    initial_conditions = [(1, 1), (2, 2), (3, 1), (1, 3)]
    
    for i, (x0, y0) in enumerate(initial_conditions):
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
        
        plt.subplot(2, 2, 1)
        plt.plot(t, x, label=f'x0={x0}, y0={y0}', linewidth=2)
        
        plt.subplot(2, 2, 2)
        plt.plot(t, y, label=f'x0={x0}, y0={y0}', linewidth=2)
        
        plt.subplot(2, 2, 3)
        plt.plot(x, y, label=f'x0={x0}, y0={y0}', linewidth=2)
    
    plt.subplot(2, 2, 1)
    plt.xlabel('Time t')
    plt.ylabel('Prey Population (x)')
    plt.title('Prey Population under Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.xlabel('Time t')
    plt.ylabel('Predator Population (y)')
    plt.title('Predator Population under Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.xlabel('Prey Population (x)')
    plt.ylabel('Predator Population (y)')
    plt.title('Phase Space Trajectories for Different Initial Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 能量守恒检验
    x0, y0 = 2, 2
    t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, x0, y0, t_span, dt)
    
    # Lotka-Volterra系统的守恒量：H = gamma*x + beta*y - delta*ln(x) - alpha*ln(y)
    H = gamma * x + beta * y - delta * np.log(x) - alpha * np.log(y)
    
    plt.subplot(2, 2, 4)
    plt.plot(t, H, 'purple', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Conserved Quantity H')
    plt.title('Energy Conservation Test')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示Lotka-Volterra模型的完整分析
    
    执行步骤：
    1. 设置参数并求解基本问题
    2. 比较不同数值方法
    3. 分析参数影响
    4. 输出数值统计结果
    """
    # 参数设置（根据题目要求）
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    x0, y0 = 2.0, 2.0
    t_span = (0, 30)
    dt = 0.01
    
    print("=== Lotka-Volterra捕食者-猎物模型分析 ===")
    print(f"参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    print(f"初始条件: x0={x0}, y0={y0}")
    print(f"时间范围: {t_span}, 步长: {dt}")
    
    try:
        # TODO: 实现主函数逻辑 (约15-20行代码)
        # 1. 基本求解
        print("\n1. 使用4阶龙格-库塔法求解...")
        # [STUDENT_CODE_HERE]
        
        # 2. 方法比较
        print("\n2. 比较不同数值方法...")
        # [STUDENT_CODE_HERE]
        
        # 3. 参数分析
        print("\n3. 分析参数影响...")
        # [STUDENT_CODE_HERE]
        
        # 4. 数值结果统计
        print("\n4. 数值结果统计:")
        # [STUDENT_CODE_HERE]
        
    except NotImplementedError as e:
        print(f"\n错误: {e}")
        print("请完成相应函数的实现后再运行主程序。")


if __name__ == "__main__":
    main()
