import numpy as np
import matplotlib.pyplot as plt

def kalman_filter_predict(x, P, a0, dt, Q):
    """
    预测步骤：
      x: 当前状态向量 [d, v_rel, a_f]
      P: 当前协方差矩阵
      a0: 本车加速度（已知输入）
      dt: 时间步长
      Q: 过程噪声协方差矩阵
    返回：预测后的状态和协方差
    """
    # 状态转移矩阵 F
    F = np.array([
        [1, dt, 0.5 * dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    # 控制输入矩阵 B（扣除本车加速度的影响）
    B = np.array([
        [-0.5 * dt**2],
        [-dt],
        [0]
    ])
    
    # 预测状态
    x_pred = F.dot(x) + B.dot(np.array([a0]))
    # 预测协方差（加入过程噪声）
    P_pred = F.dot(P).dot(F.T) + Q
    return x_pred, P_pred

def kalman_filter_update(x_pred, P_pred, z, H, R):
    """
    更新步骤：
      x_pred: 预测状态向量
      P_pred: 预测协方差矩阵
      z: 测量值（相对距离，1 维向量）
      H: 测量矩阵
      R: 测量噪声协方差
    返回：更新后的状态和协方差
    """
    # 计算测量残差
    y = z - H.dot(x_pred)
    # 残差协方差 S （标量）
    S = H.dot(P_pred).dot(H.T) + R
    # 卡尔曼增益（形状为 (3,1)）
    K = P_pred.dot(H.T) / S
    # 更新状态向量
    x_new = x_pred + K.flatten() * y
    # 更新协方差矩阵
    P_new = (np.eye(len(x_pred)) - K.dot(H)).dot(P_pred)
    return x_new, P_new

def simulate_system(T=20.0, dt=0.1):
    """
    仿真真实系统状态：
      - 假设前车加速度采用一个正弦信号（例如 a_f_true = 2*sin(0.3*t)）
      - 本车加速度 a0 取 0（可根据需要调整）
      - 初始相对距离为 30 m，初始相对速度为 0
    同时根据前车真实状态生成相机测量（距离，加上噪声）
    
    返回：
      t: 时间向量
      a_f_true: 真实前车加速度数组
      d_true: 真实相对距离数组
      v_rel_true: 真实相对速度数组
      z_meas: 模拟的距离测量（添加了测量噪声）
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    
    # 真实前车加速度（使用正弦函数模拟变化）
    a_f_true = 2.0 * np.sin(0.3 * t)
    
    # 设定本车加速度 a0（在此取 0，可根据实际情况设定）
    a0 = 0.0
    
    # 初始化真实状态
    d_true = np.zeros(n_steps)
    v_rel_true = np.zeros(n_steps)
    
    d_true[0] = 30.0  # 初始相对距离 30 m
    v_rel_true[0] = 0.0  # 初始相对速度 0 m/s
    
    for k in range(n_steps - 1):
        # 根据动力学方程更新真实状态
        # d[k+1] = d[k] + v_rel[k]*dt + 0.5*(a_f_true[k] - a0)*dt^2
        d_true[k+1] = d_true[k] + v_rel_true[k] * dt + 0.5 * (a_f_true[k] - a0) * dt**2
        # v_rel[k+1] = v_rel[k] + (a_f_true[k] - a0)*dt
        v_rel_true[k+1] = v_rel_true[k] + (a_f_true[k] - a0) * dt
    
    # 模拟相机测量：观测为相对距离，加入高斯噪声
    # 假设测量噪声方差为 R_true = 2.0
    R_true = 2.0
    noise = np.random.normal(0, np.sqrt(R_true), size=n_steps)
    z_meas = d_true + noise
    
    return t, a_f_true, d_true, v_rel_true, z_meas, a0

def run_kalman_filter(t, z_meas, a0, dt, Q, R, P0):
    """
    利用卡尔曼滤波器处理测量数据，估计状态变化
      这里只更新状态向量 [d, v_rel, a_f]
    返回：
      a_f_kf: 每步滤波得到的前车加速度估计值
    """
    n_steps = len(t)
    # 状态向量初始值：[d, v_rel, a_f]
    # 初始可取与真实状态接近或者直接设定为 0（这里设初始 d 与测量接近）
    x = np.array([z_meas[0], 0.0, 0.0])
    P = P0.copy()
    
    # 测量矩阵 H：仅观测相对距离 d
    H = np.array([[1, 0, 0]])
    
    # 存储滤波器输出的前车加速度估计
    a_f_kf = np.zeros(n_steps)
    
    for k in range(n_steps):
        # 预测步骤
        x_pred, P_pred = kalman_filter_predict(x, P, a0, dt, Q)
        # 测量
        z = np.array([z_meas[k]])
        # 更新步骤
        x, P = kalman_filter_update(x_pred, P_pred, z, H, R)
        # 存储前车加速度估计（状态向量的第三个分量）
        a_f_kf[k] = x[2]
    
    return a_f_kf

def main():
    # 仿真参数
    dt = 0.01            # 时间步长（秒）
    T = 20.0            # 总仿真时间（秒）
    
    # 仿真真实系统
    t, a_f_true, d_true, v_rel_true, z_meas, a0 = simulate_system(T, dt)
    
    # 卡尔曼滤波器参数（这些需要根据实际情况进行调节）
    # 过程噪声协方差 Q：反映模型动态的不确定性
    Q = np.array([
        [0.001, 0, 0],
        [0, 0.001, 0],
        [0, 0, 0.001]
    ])
    # 测量噪声协方差 R（与 simulate_system 中的 R_true 相对应）
    R = 0.1
    # 初始协方差矩阵 P0
    P0 = np.eye(3) * 1.0
    
    # 运行卡尔曼滤波
    a_f_kf = run_kalman_filter(t, z_meas, a0, dt, Q, R, P0)
    
    # 绘制真实前车加速度与滤波估计前车加速度的对比图
    plt.figure(figsize=(10, 5))
    plt.plot(t, a_f_true, label='True lead car acceleration', linewidth=2)
    plt.plot(t, a_f_kf, label='estimated acceleration', linestyle='--')
    plt.xlabel('time(s)')
    plt.ylabel('acceleration(m/s²)')
    plt.title('True vs Estimated Lead Car Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
