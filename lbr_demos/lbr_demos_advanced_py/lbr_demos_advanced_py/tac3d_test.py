import numpy as np
import math
import time
import cv2
import json
import pickle
from .contact_analysis import estimate_rotation_center

ROTATION_LEFT_GT = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)
ROTATION_RIGHT_GT = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)


def gripper_force_direct(force_left, force_right, matrix_left, matrix_right):
    # F_y = (force_right[0,0] - force_left[0,0]) / 2
    # F_x = (matrix_right[0,2] - matrix_left[0,2]) / 2
    # F_z = -(force_left[0,0] + force_right[0,0]) / 2
    # T_z = -(force_left[0,1] + force_right[0,1]) / 2

    F_x = -force_right[0, 1] + force_left[0, 1] #夹爪坐标系中X方向受力
    F_y = -force_left[0, 2] + force_right[0, 2] #夹爪坐标系中Y方向受力
    F_z = -(force_left[0,0] + force_right[0,0]) #夹爪坐标系中Z方向受力
    R_x = -matrix_right[0,1] + matrix_left[0,1] #夹爪坐标系中X轴力矩(未校准)
    R_y = -matrix_left[0,2] + matrix_right[0,2] #夹爪坐标系中Y轴力矩（未校准）
    R_z = -(matrix_left[0, 0] + matrix_right[0,0]) #夹爪坐标系中Z轴力矩（未校准）

    return F_x, F_y, F_z, R_x, R_y, R_z

def gripper_force_only(force_left, force_right):
    # F_y = (force_right[0,0] - force_left[0,0]) / 2
    # F_x = (matrix_right[0,2] - matrix_left[0,2]) / 2
    # F_z = -(force_left[0,0] + force_right[0,0]) / 2
    # T_z = -(force_left[0,1] + force_right[0,1]) / 2

    F_x = -force_right[0, 1] + force_left[0, 1] #夹爪坐标系中X方向受力
    F_y = -force_left[0, 2] + force_right[0, 2] #夹爪坐标系中Y方向受力
    F_z = -(force_left[0,0] + force_right[0,0]) #夹爪坐标系中Z方向受力

    return F_x, F_y, F_z


def cal_force(flow,
              delta):  # claculate the trendlines' vector sum to represent the force in the paralleral and the torque of the z axis.
    force_x = 0
    force_y = 0
    torque = 0  # the center when calculating the torque should be reset at the center of the object.
    force_x = np.sum(flow[5:15, 5:15, 0])
    force_y = np.sum(flow[5:15, 5:15, 1])
    force_z = np.sum(flow[5:15, 5:15, 2])
    torque = np.mean(np.gradient(flow[:, :, 1])[1] - np.gradient(flow[:, :, 0])[0]) / delta

    return force_x, force_y, force_z, torque

def cal_div_curl(v_field, delta):
    div = np.mean(np.gradient(v_field[:, :, 0])[1] + np.gradient(v_field[:, :, 1])[0]) / delta
    curl = np.mean(np.gradient(v_field[:, :, 1])[1] - np.gradient(v_field[:, :, 0])[0]) / delta

    return div, curl

def gripper_force_flow(flow_left, flow_right, delta):
    # 左侧传感器受力情况
    force_x_left, force_y_left, force_z_left, torque_left = cal_force(flow_left, delta)
    div_left, curl_left = cal_div_curl(flow_left, delta)
    # 右侧传感器受力情况
    force_x_right, force_y_right, force_z_right, torque_right = cal_force(flow_right, delta)
    div_right, curl_right = cal_div_curl(flow_right, delta)
    # # 计算夹爪受力情况(原坐标系)
    # F_x = -(force_x_left + force_x_right) * 1
    # F_y = (curl_left - curl_right) / 2
    # F_z = (force_x_left - force_x_right) / 2
    # T_z = (force_y_left - force_y_right) * 1

    # # 计算夹爪受力情况(机械臂坐标系)
    # F_y = (force_x_right - force_x_left) / 2
    # F_x = (curl_right - curl_left) / 2
    # F_z = -(force_x_left + force_x_right) / 2
    # T_z = -(force_y_left + force_y_right) / 2

    r = 6
    F_x = -force_y_right + force_y_left  # 夹爪坐标系中X方向受力
    F_y = -force_z_left + force_z_right  # 夹爪坐标系中Y方向受力
    F_z = -(force_x_left + force_x_right)  # 夹爪坐标系中Z方向受力
    R_x = (force_x_right - force_x_left) * r  # 夹爪坐标系中X轴力矩
    R_y = (curl_right - curl_left)  # 夹爪坐标系中Y轴力矩
    R_z = -(force_y_left + force_y_right) * r  # 夹爪坐标系中Z轴力矩

    return F_x, F_y, F_z, R_x, R_y, R_z

def skew_symmetric(f):
    fx, fy, fz = f[0,0], f[0,1], f[0,2]
    return np.array([
        [0, fz, -fy],
        [-fz, 0, fx],
        [fy, -fx, 0]
    ])

def contact_point_estimate(FL, FR, ML, MR, FC, delta_L, delta_R, z0=-150.0):
    """

    需要进一步优化

    """
    skew_FL = skew_symmetric(FL)
    skew_FR = skew_symmetric(FR)
    skew_FC = skew_symmetric(FC)
    right = ML + skew_FL @ delta_L + MR + skew_FR @ delta_R
    S = skew_FC
    m = right.T.reshape(3,)
    A = S[:, :2]  # shape (3,2)
    b = m - S[:, 2] * z0  # shape (3,)

    # 检查 A 的秩（是否能唯一确定 x,y）
    rank = np.linalg.matrix_rank(A, tol=None)
    if rank < 2:
        # 退化情况：无法用唯一的 x,y 确定（例如力方向导致列线性相关）
        print("线性系统退化 (rank < 2)，且无一致解 —— 不能唯一确定 (x,y)。")
        return np.array([0.0, 0.0, float(z0)])
    if np.linalg.norm(FC) < 0.05:
        return np.array([0.0, 0.0, float(z0)])
    # 正常情况：A 满秩，用最小二乘（实际上为精确解）
    sol, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=None)
    x, y = float(sol[0]), float(sol[1])
    return np.array([x, y, float(z0)])
    
def visualize_tactile_image(tactile_array, tactile_resolution=30, scale = 10,
                            normal_force_threshold=0.0008):
    resolution = tactile_resolution
    T = len(tactile_array)
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_array[row, col][0] * scale
            loc1_y = loc0_y + tactile_array[row, col][1] * scale

            cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), (0, 255, 255), 2)

    return imgs_tactile

def estimate_rigid_transform_kabsch(A, B):
    """
    A, B: (N,3) numpy arrays of corresponding points.
    Returns R (3x3), t (3,)
    (standard Kabsch / Umeyama without scale)
    """
    assert A.shape == B.shape and A.shape[1] == 3
    N = A.shape[0]
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def select_top_n(matrix, N=10, by_abs=False):
    """
    从 matrix（shape=(400,3)）中提取最大的 N 个值的索引，并绘制三维散点图，
    高亮这 N 个点。

    :param matrix: numpy.ndarray, shape (400, 3)
    :param N: int, 需突出显示的最大值数量
    :param by_abs: bool, 是否按绝对值排序
    """
    assert matrix.ndim == 2 and matrix.shape[1] == 3, "输入应为 shape=(400,3)"

    flat = matrix.flatten()  # 展平 (1200,)
    if by_abs:
        idx_flat = np.argpartition(np.abs(flat), -N)[-N:]
    else:
        idx_flat = np.argpartition(flat, -N)[-N:]
        # 若需按值排序可: idx_flat = idx_flat[np.argsort(flat[idx_flat])]

    # 将展平索引转换为 matrix 的行列索引
    row_idx = idx_flat // 3
    col_idx = idx_flat % 3

    # # 获取顶 N 点的坐标
    # top_points = matrix_1[row_idx, :]  # shape (N,3)

    return row_idx, col_idx

# # 创建Sensor实例，设置回调函数为上面写好的Tac3DRecvCallback，设置UDP接收端口为9988，数据帧缓存队列最大长度为5
# tac3d_right = PyTac3D.Sensor(recvCallback=Tac3DRecvCallback, port=9988)
# tac3d_left = PyTac3D.Sensor(recvCallback=Tac3DRecvCallback, port=9989)
#
# # 等待Tac3D-Desktop端启动传感器并建立连接
# tac3d_left.waitForFrame()
# tac3d_right.waitForFrame()
#
# time.sleep(5) # 5s
#
# # 发送一次校准信号（应确保校准时传感器未与任何物体接触！否则会输出错误的数据！）
# # tac3d_left.calibrate(SN)
# # tac3d_right.calibrate(SN)
#
# time.sleep(5) #5s
# t1 = time.time()
#
# # 获取frame的另一种方式：通过getFrame获取缓存队列中的frame
# frame_left = tac3d_left.getFrame()
# frame_right = tac3d_right.getFrame()
# while True:
#     frame_left = tac3d_left.getFrame()
#     frame_right = tac3d_right.getFrame()
#     if frame_left is not None and frame_right is not None:
#         force_left, force_right = frame_left['3D_ResultantForce'], frame_right['3D_ResultantForce']
#         matrix_left, matrix_right = frame_left['3D_ResultantMoment'], frame_right['3D_ResultantMoment']
#         print('force_left=', force_left)
#         print('force_right=', force_right)
#
#         fordis_left, fordis_right = frame_left['3D_Forces'].reshape(20, 20, 3), frame_right['3D_Forces'].reshape(20, 20, 3)
#     else:
#         print('force_left=', force_left)
#         print('force_right=', force_right)
#
#     Force_left.append(force_left)
#     Force_right.append(force_right)
#     Matrix_left.append(matrix_left)
#     Matrix_right.append(matrix_right)
#     Fordis_left.append(fordis_left)
#     Fordis_right.append(fordis_right)
#
#     # 图形化表示
#     tactile_left_image = visualize_tactile_image(fordis_left)
#     tactile_right_image = visualize_tactile_image(fordis_right)
#     cv2.imshow('tactile_left', tactile_left_image)
#     cv2.imshow('tactile_right', tactile_right_image)
#     cv2.waitKey(1)
#
#     ##计算夹持物体的运动趋势（使用两种方法）
#     f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(force_left, force_right, matrix_left, matrix_right)
#     f_x_1, f_y_1, f_z_1, r_x_1, r_y_1, r_z_1 = gripper_force_flow(flow_left=fordis_left, flow_right=fordis_right, delta=1 / 20)
#     F_x.append(f_x)
#     F_y.append(f_y)
#     F_z.append(f_z)
#     R_x.append(r_x), R_y.append(r_y), R_z.append(r_z)
#     F_x_1.append(f_x_1)
#     F_y_1.append(f_y_1)
#     F_z_1.append(f_z_1)
#     R_x_1.append(r_x_1), R_y_1.append(r_y_1), R_z_1.append(r_z_1)
#
#     time.sleep(0.05) #0.s
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#     # t2 = time.time()
#     # if t2-t1 > 10:
#     #     break
# cv2.destroyAllWindows()
# ##保存数据
# np.savez(r'Force_data.npz', Force_left=Force_left, Force_right=Force_right,
#          Matrix_left=Matrix_left, Matrix_right=Matrix_right,
#          Fordis_left=Fordis_left, Fordis_right=Fordis_right,
#          F_x=F_x, F_y=F_y, F_z=F_z, R_x=R_x, R_y=R_y, R_z=R_z,
#          F_x_1=F_x_1, F_y_1 = F_y_1, F_z_1 = F_z_1, R_x_1 = R_x_1, R_y_1 = R_y_1, R_z_1 = R_z_1)
# # # 发送一次关闭传感器的信号（不建议使用）
# # tac3d.quitSensor(SN)
