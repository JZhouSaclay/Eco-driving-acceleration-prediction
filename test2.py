import numpy as np
import matplotlib.pyplot as plt

def simulate_system(T=20.0, dt=0.1):
    """
    仿真真实系统状态：
      - 前车加速度 a_f_true = 2*sin(0.3*t)
      - 本车加速度 a0 = 0
      - 仅得到相对距离测量 z_meas（含噪声）
    返回:
      t, a_f_true, d_true, v_rel_true, z_meas, a0
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps, endpoint=False)
    
    # 真实前车加速度（正弦）
    a_f_true = 2.0 * np.sin(0.3 * t)
    
    # 本车加速度(设为0)
    a0 = 0.0
    
    # 真实相对距离、相对速度
    d_true = np.zeros(n_steps)
    v_rel_true = np.zeros(n_steps)
    
    # 初值
    d_true[0] = 30.0
    v_rel_true[0] = 0.0
    
    # 根据简单离散动力学进行状态演化
    for k in range(n_steps - 1):
        d_true[k+1] = d_true[k] + v_rel_true[k] * dt + 0.5 * (a_f_true[k] - a0) * dt**2
        v_rel_true[k+1] = v_rel_true[k] + (a_f_true[k] - a0) * dt
    
    # 加测量噪声
    R_true = 2.0  # 测距噪声方差
    noise = np.random.normal(0, np.sqrt(R_true), n_steps)
    z_meas = d_true + noise
    
    return t, a_f_true, d_true, v_rel_true, z_meas, a0


def kalman_filter_predict(x, P, a0, dt, Q):
    """
    KF 预测
    x: [d, v_rel, a_f]
    """
    F = np.array([
        [1, dt, 0.5*dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    B = np.array([
        [-0.5*dt**2],
        [-dt],
        [0]
    ])
    # 预测
    x_pred = F.dot(x) + B.dot([a0])
    P_pred = F.dot(P).dot(F.T) + Q
    return x_pred, P_pred

def kalman_filter_update(x_pred, P_pred, z, H, R):
    """
    KF 更新
    z: 相机测量的相对距离
    """
    y = z - H.dot(x_pred)      # 测量残差
    S = H.dot(P_pred).dot(H.T) + R  # 残差协方差(标量)
    K = P_pred.dot(H.T) / S         # 卡尔曼增益 (3x1)
    x_new = x_pred + K.flatten() * y
    P_new = (np.eye(3) - K.dot(H)).dot(P_pred)
    return x_new, P_new

def run_kalman_filter(t, z_meas, a0, dt, Q, R, P0):
    n_steps = len(t)
    # 状态初始 [d, v_rel, a_f]
    x = np.array([z_meas[0], 0.0, 0.0])
    P = P0.copy()
    H = np.array([[1, 0, 0]])
    
    # 存储 KF 输出
    a_f_kf = np.zeros(n_steps)
    
    for k in range(n_steps):
        # 预测
        x_pred, P_pred = kalman_filter_predict(x, P, a0, dt, Q)
        # 更新
        z = np.array([z_meas[k]])
        x, P = kalman_filter_update(x_pred, P_pred, z, H, R)
        a_f_kf[k] = x[2]  # 第三维为前车加速度
    return a_f_kf

def exponential_weighted_moving_avg(data, window=6, alpha=0.95):
    """
    对 data 做一个简单的指数加权滑动平均 (EWMA):
      - 取最近的 window 个数据
      - 距当前越近的数权重越大(以 alpha^j 递减)
      - 归一化后得到平滑值
    """
    n = len(data)
    ewm = np.zeros(n)
    for i in range(n):
        sum_w = 0.0
        sum_val = 0.0
        # 向后看 window 个数据
        for j in range(window):
            idx = i - j
            if idx < 0:
                break
            w = alpha**j       # 指数权重
            sum_w += w
            sum_val += w * data[idx]
        # 如果 i 太小还没够 window 个，就只用可用数据
        ewm[i] = sum_val / sum_w if sum_w > 1e-12 else data[i]
    return ewm

def main():
    dt = 0.01
    T = 20.0
    t, a_f_true, d_true, v_rel_true, z_meas, a0 = simulate_system(T, dt)
    
    # KF 参数
    Q = np.array([
        [0.001, 0,    0   ],
        [0,     0.001,0   ],
        [0,     0,    0.001]
    ])
    R = 0.1
    P0 = np.eye(3) * 1.0
    
    # 运行 KF
    a_f_kf = run_kalman_filter(t, z_meas, a0, dt, Q, R, P0)
    
    # 对真实加速度做 EWMA 平滑(仅作对比)
    a_f_ewma = exponential_weighted_moving_avg(a_f_true, window=6, alpha=0.95)
    
    # 画图对比
    plt.figure(figsize=(10, 5))
    plt.plot(t, a_f_true, label='True lead car acceleration', linewidth=2)
    plt.plot(t, a_f_kf, '--', label='KF estimated acceleration')
    plt.plot(t, a_f_ewma, '-.', label='EWMA (window=6, alpha=0.95)')
    
    plt.xlabel('time(s)')
    plt.ylabel('acceleration(m/s²)')
    plt.title('True vs Estimated Lead Car Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
