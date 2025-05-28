#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：洛伦兹方程学生模板
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


def lorenz_system(state, sigma, r, b):
    """
    定义洛伦兹系统方程
    
    参数:
        state: 当前状态向量 [x, y, z]
        sigma, r, b: 系统参数
        
    返回:
        导数向量 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])

def lorenz_system_for_solve_ivp(t, state, sigma, r, b):
    """
    为solve_ivp适配的洛伦兹系统方程
    """
    return lorenz_system(state, sigma, r, b)

def solve_lorenz_equations(sigma=10.0, r=28.0, b=8/3,
                          x0=0.1, y0=0.1, z0=0.1,
                          t_span=(0, 50), dt=0.01):
    """
    求解洛伦兹方程
    
    返回:
        t: 时间点数组
        y: 解数组，形状为(3, n_points)
    """
    # 设置初始状态
    initial_state = np.array([x0, y0, z0])
    
    # 设置时间点
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/dt))
    
    # 求解微分方程
    sol = solve_ivp(lorenz_system_for_solve_ivp, 
                    t_span=t_span, 
                    y0=initial_state,
                    t_eval=t_eval, 
                    method='RK45',
                    args=(sigma, r, b))
    
    return sol.t, sol.y
                              
def plot_lorenz_attractor(t: np.ndarray, y: np.ndarray):
    """
    绘制洛伦兹吸引子3D图
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D轨迹
    ax.plot(y[0], y[1], y[2], lw=0.5, color='blue')
    
    # 设置标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Lorenz Attractor')
    
    plt.tight_layout()
    plt.show()
    
def compare_initial_conditions(ic1, ic2, t_span=(0, 50), dt=0.01):
    """
    比较不同初始条件的解
    """
    # 求解两个初始条件的解
    t1, y1 = solve_lorenz_equations(x0=ic1[0], y0=ic1[1], z0=ic1[2], t_span=t_span, dt=dt)
    t2, y2 = solve_lorenz_equations(x0=ic2[0], y0=ic2[1], z0=ic2[2], t_span=t_span, dt=dt)
    
    # 绘制x分量随时间变化对比
    plt.figure(figsize=(12, 6))
    plt.plot(t1, y1[0], label=f'IC1: {ic1}', alpha=0.7)
    plt.plot(t2, y2[0], label=f'IC2: {ic2}', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('x Value')
    plt.title('x(t) Comparison with Different Initial Conditions')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 计算并绘制相空间距离
    distance = np.sqrt((y1[0]-y2[0])**2 + (y1[1]-y2[1])**2 + (y1[2]-y2[2])**2)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(t1, distance)
    plt.xlabel('Time')
    plt.ylabel('Phase Space Distance (log scale)')
    plt.title('Trajectory Separation in Phase Space')
    plt.grid()
    plt.show()

def main():
    """
    主函数，执行所有任务
    """
    # 任务A: 求解洛伦兹方程
    t, y = solve_lorenz_equations()
    
    # 任务B: 绘制洛伦兹吸引子
    plot_lorenz_attractor(t, y)
    
    # 任务C: 比较不同初始条件
    ic1 = (0.1, 0.1, 0.1)
    ic2 = (0.10001, 0.1, 0.1)  # 微小变化
    compare_initial_conditions(ic1, ic2)


if __name__ == '__main__':
    main()
