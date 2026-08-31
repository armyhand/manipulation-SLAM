import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

"""
20260607:下一步拟进行联合优化，通过与姿态偏转结合起来，以缩小marker点的位移作为目标判断接触点。
即有两个约束，一个是力和力矩的约束，这个约束的权重应该较弱；另一个约束是偏转marker变化约束。
"""

def select_top_n(matrix, matrix_1, N=10, by_abs=False):
    """
    从 matrix（shape=(400,3)）中提取最大的 N 个值的索引，并绘制三维散点图，
    高亮这 N 个点。

    :param matrix: numpy.ndarray, shape (400, 3)
    :param N: int, 需突出显示的最大值数量
    :param by_abs: bool, 是否按绝对值排序
    """
    assert matrix.ndim == 2 and matrix.shape[1] == 3, "输入应为 shape=(400,3)"

    ## 我们只关心Z方向
    matrix = matrix[:, 2]  ##(300)
    flat = matrix.flatten()  # 展平 (1200,)
    if by_abs:
        idx_flat = np.argpartition(np.abs(flat), -N)[-N:]
    else:
        idx_flat = np.argpartition(flat, -N)[-N:]
        # 若需按值排序可: idx_flat = idx_flat[np.argsort(flat[idx_flat])]

    # 将展平索引转换为 matrix 的行列索引
    row_idx = idx_flat
    # 获取顶 N 点的坐标
    top_points = matrix_1[row_idx, :]  # shape (N,3)

    return top_points


def plot_top_n(matrix, matrix_1, N=10, by_abs=False):
    """
    从 matrix（shape=(400,3)）中提取最大的 N 个值的索引，并绘制三维散点图，
    高亮这 N 个点。

    :param matrix: numpy.ndarray, shape (400, 3)
    :param N: int, 需突出显示的最大值数量
    :param by_abs: bool, 是否按绝对值排序
    """
    assert matrix.ndim == 2 and matrix.shape[1] == 3, "输入应为 shape=(400,3)"
    ## 我们只关心Z方向
    matrix = matrix[:, 2]  ##(300)
    flat = matrix.flatten()  # 展平 (1200,)
    if by_abs:
        idx_flat = np.argpartition(np.abs(flat), -N)[-N:]
    else:
        idx_flat = np.argpartition(flat, -N)[-N:]
        # 若需按值排序可: idx_flat = idx_flat[np.argsort(flat[idx_flat])]

    # 将展平索引转换为 matrix 的行列索引
    row_idx = idx_flat

    # 获取顶 N 点的坐标
    top_points = matrix_1[row_idx, :]  # shape (N,3)
    xs, ys, zs = top_points[:, 0], top_points[:, 1], top_points[:, 2]

    return xs, ys, zs

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

def select_top_n_indices(matrix, N=10, by_abs=False, component=2):
    """
    返回位移最显著的 N 个 marker 索引。

    默认使用 z 方向位移作为活跃区域判据，与原脚本保持一致。
    """
    assert matrix.ndim == 2 and matrix.shape[1] == 3, "输入应为 shape=(400,3)"
    values = matrix[:, component]
    scores = np.abs(values) if by_abs else values
    return np.argpartition(scores, -N)[-N:]

def rotation_angle_from_matrix(R):
    """由旋转矩阵求旋转角，返回弧度。"""
    trace_value = (np.trace(R) - 1.0) / 2.0
    trace_value = np.clip(trace_value, -1.0, 1.0)
    return np.arccos(trace_value)

def solve_rotation_center_from_transforms(
    rotations, translations, anchor_point, regularization=1e-3
):
    """
    根据多个刚体变换联合求解一个稳定的旋转中心点。

    对每个刚体变换满足:
        p' = R p + t
    若旋转中心为 c，则理想情况下:
        c = R c + t
        (I - R) c = t

    由于噪声和沿旋转轴方向的自由度，上式一般用带正则项的最小二乘求解，
    并使用 anchor_point 约束解落在接触区域附近。
    """
    if len(rotations) == 0:
        return anchor_point.copy()

    system_matrix = []
    system_rhs = []
    for R, t, weight in zip(rotations, translations, np.ones(len(rotations))):
        scale = np.sqrt(weight)
        system_matrix.append(scale * (np.eye(3) - R))
        system_rhs.append(scale * t)

    A = np.vstack(system_matrix)
    b = np.concatenate(system_rhs)
    lhs = A.T @ A + regularization * np.eye(3)
    rhs = A.T @ b + regularization * anchor_point
    return np.linalg.solve(lhs, rhs)

def estimate_rotation_center_series(
    position_left,
    position_right,
    diff_position_left,
    diff_position_right,
    rotation_left,
    rotation_right,
    dis_left,
    dis_right,
    top_n=30,
    min_rotation_deg=0.5,
    min_motion_mm=0.02,
    regularization=1e-3,
):
    """
    根据 marker 位移估计每一帧的旋转中心（世界坐标系）。

    实现步骤：
    1. 在左右传感器上分别选择位移最大的 marker 作为活跃接触区域；
    2. 用这些 marker 的初始位置和当前位姿做 Kabsch 刚体配准；
    3. 联合左右两侧刚体变换求解共享的旋转中心；
    4. 当旋转角或位移过小时，回退到上一帧结果以增强稳定性。
    """
    total_frames = len(position_left)
    min_rotation_rad = np.deg2rad(min_rotation_deg)
    rotation_centers = np.zeros((total_frames, 3))

    left_initial_world = (rotation_left @ position_left[0].T).T + dis_left
    right_initial_world = (rotation_right @ position_right[0].T).T + dis_right
    rotation_centers[0] = (
        left_initial_world.mean(axis=0) + right_initial_world.mean(axis=0)
    ) / 2.0

    left_indices = select_top_n_indices(
        diff_position_left[10], N=top_n, by_abs=True, component=2
    )
    right_indices = select_top_n_indices(
        diff_position_right[10], N=top_n, by_abs=True, component=2
    )

    for t in range(total_frames):
        rotations = []
        translations = []
        anchor_candidates = []

        # left_indices = select_top_n_indices(
        #     diff_position_left[t], N=top_n, by_abs=True, component=2
        # )
        # right_indices = select_top_n_indices(
        #     diff_position_right[t], N=top_n, by_abs=True, component=2
        # )

        left_current_world = (rotation_left @ position_left[t].T).T + dis_left
        right_current_world = (rotation_right @ position_right[t].T).T + dis_right

        left_initial_active = left_initial_world[left_indices]
        left_current_active = left_current_world[left_indices]
        right_initial_active = right_initial_world[right_indices]
        right_current_active = right_current_world[right_indices]

        R_left_active, t_left_active = estimate_rigid_transform_kabsch(
            left_initial_active, left_current_active
        )
        R_right_active, t_right_active = estimate_rigid_transform_kabsch(
            right_initial_active, right_current_active
        )

        left_rotation_angle = rotation_angle_from_matrix(R_left_active)
        right_rotation_angle = rotation_angle_from_matrix(R_right_active)
        left_motion = np.mean(
            np.linalg.norm(left_current_active - left_initial_active, axis=1)
        )
        right_motion = np.mean(
            np.linalg.norm(right_current_active - right_initial_active, axis=1)
        )

        if left_rotation_angle >= min_rotation_rad and left_motion >= min_motion_mm:
            rotations.append(R_left_active)
            translations.append(t_left_active)
            anchor_candidates.append(left_current_active.mean(axis=0))

        if right_rotation_angle >= min_rotation_rad and right_motion >= min_motion_mm:
            rotations.append(R_right_active)
            translations.append(t_right_active)
            anchor_candidates.append(right_current_active.mean(axis=0))

        if len(anchor_candidates) == 0:
            if t > 0:
                rotation_centers[t] = rotation_centers[t - 1]
            continue

        anchor_point = np.mean(anchor_candidates, axis=0)
        rotation_centers[t] = solve_rotation_center_from_transforms(
            rotations=rotations,
            translations=translations,
            anchor_point=anchor_point,
            regularization=regularization,
        )

    return rotation_centers

def estimate_rotation_center(
    position_left,
    position_right,
    diff_position_left,
    diff_position_right,
    rotation_left,
    rotation_right,
    top_n=30,
    min_rotation_deg=0.5,
    min_motion_mm=0.02,
    regularization=1e-3,
):
    """
    根据 marker 位移估计每一帧的旋转中心（世界坐标系）。

    实现步骤：
    1. 在左右传感器上分别选择位移最大的 marker 作为活跃接触区域；
    2. 用这些 marker 的初始位置和当前位姿做 Kabsch 刚体配准；
    3. 联合左右两侧刚体变换求解共享的旋转中心；
    4. 当旋转角或位移过小时，回退到上一帧结果以增强稳定性。
    """
    total_frames = len(position_left)
    min_rotation_rad = np.deg2rad(min_rotation_deg)
    rotation_centers_left = np.zeros((total_frames, 3))
    rotation_centers_right = np.zeros((total_frames, 3))

    left_initial_world = (rotation_left @ position_left[0].T).T
    right_initial_world = (rotation_right @ position_right[0].T).T
    rotation_centers_left[0] = left_initial_world.mean(axis=0)
    rotation_centers_right[0] = right_initial_world.mean(axis=0)

    left_indices = select_top_n_indices(
        diff_position_left[10], N=top_n, by_abs=True, component=2
    )
    right_indices = select_top_n_indices(
        diff_position_right[10], N=top_n, by_abs=True, component=2
    )

    for t in range(total_frames):
        rotations_left = []
        rotations_right = []
        translations_left = []
        translations_right = []
        anchor_candidates_left = []
        anchor_candidates_right = []

        # left_indices = select_top_n_indices(
        #     diff_position_left[t], N=top_n, by_abs=True, component=2
        # )
        # right_indices = select_top_n_indices(
        #     diff_position_right[t], N=top_n, by_abs=True, component=2
        # )

        left_current_world = (rotation_left @ position_left[t].T).T
        right_current_world = (rotation_right @ position_right[t].T).T

        left_initial_active = left_initial_world[left_indices]
        left_current_active = left_current_world[left_indices]
        right_initial_active = right_initial_world[right_indices]
        right_current_active = right_current_world[right_indices]

        R_left_active, t_left_active = estimate_rigid_transform_kabsch(
            left_initial_active, left_current_active
        )
        R_right_active, t_right_active = estimate_rigid_transform_kabsch(
            right_initial_active, right_current_active
        )

        left_rotation_angle = rotation_angle_from_matrix(R_left_active)
        right_rotation_angle = rotation_angle_from_matrix(R_right_active)
        left_motion = np.mean(
            np.linalg.norm(left_current_active - left_initial_active, axis=1)
        )
        right_motion = np.mean(
            np.linalg.norm(right_current_active - right_initial_active, axis=1)
        )

        if left_rotation_angle >= min_rotation_rad and left_motion >= min_motion_mm:
            rotations_left.append(R_left_active)
            translations_left.append(t_left_active)
            anchor_candidates_left.append(left_current_active.mean(axis=0))

        if right_rotation_angle >= min_rotation_rad and right_motion >= min_motion_mm:
            rotations_right.append(R_right_active)
            translations_right.append(t_right_active)
            anchor_candidates_right.append(right_current_active.mean(axis=0))

        if len(anchor_candidates_left) != 0:
            rotation_centers_left[t] = solve_rotation_center_from_transforms(
                rotations=rotations_left,
                translations=translations_left,
                anchor_point=left_initial_world.mean(axis=0),
                regularization=regularization,
            )
        else:
            rotation_centers_left[t] = rotation_centers_left[t-1]

        if len(anchor_candidates_right) != 0:
            rotation_centers_right[t] = solve_rotation_center_from_transforms(
                rotations=rotations_right,
                translations=translations_right,
                anchor_point=right_initial_world.mean(axis=0),
                regularization=regularization,
            )
        else:
            rotation_centers_right[t] = rotation_centers_right[t-1]

    return rotation_centers_left, rotation_centers_right


if __name__ == "__main__":
    data = np.load(
        f"Force_data_cal20260521_+x_x1y11.npz"
    )  ## 力和力矩测试数据集，表示受力点在夹爪x轴正向，受力点为（1,11）
    F_x = data["F_x"]
    F_y = data["F_y"]
    F_z = data["F_z"]  ## F_x,F_y,F_z均为世界坐标系下的合力分量
    Position_left = data["Position_left"].reshape(-1, 400, 3)
    Position_right = data["Position_right"].reshape(
        -1, 400, 3
    )  ## Position_left,Position_right均为传感器坐标系下的marker点分布位置信息，需转换到世界坐标系下进行分析
    Diff_position_left = data["Displacement_left"].reshape(-1, 400, 3)
    Diff_position_right = data["Displacement_right"].reshape(
        -1, 400, 3
    )  ## Diff_position_left,Diff_position_right均为传感器坐标系下的marker点分布位移信息，需转换到世界坐标系下进行分析
    Fordis_left = data["Fordis_left"].reshape(-1, 400, 3)
    Fordis_right = data["Fordis_right"].reshape(
        -1, 400, 3
    )  ## Fordis_left,Fordis_right均为传感器坐标系下的marker点分布力信息，需转换到世界坐标系下进行分析

    F_x = F_x - np.mean(F_x[0:10], axis=0)
    F_y = F_y - np.mean(F_y[0:10], axis=0)
    F_z = F_z - np.mean(F_z[0:10], axis=0)  ## 对合力进行基线校正，去除初始偏置
    Fordis_left = Fordis_left - np.mean(Fordis_left[0:10], axis=0)
    Fordis_right = Fordis_right - np.mean(Fordis_right[0:10], axis=0)

    ##-------------------------------------合力信息---------------------------------##
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(F_x)
    plt.title("F_x")
    plt.subplot(3, 1, 2)
    plt.plot(F_y)
    plt.title("F_y")
    plt.subplot(3, 1, 3)
    plt.plot(F_z)
    plt.title("F_z")
    plt.show()

    ##----------------------基于 marker 位移估计旋转中心--------------------------------------------##
    rotation_left = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    rotation_right = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    number = 10  ##以number的点为参考，选择位姿的估计点

    rotation_center_world_left, rotation_center_world_right = estimate_rotation_center(
        Position_left,
        Position_right,
        Diff_position_left,
        Diff_position_right,
        rotation_left,
        rotation_right,
        top_n=number,
        min_rotation_deg=0.5,
        min_motion_mm=0.02,
        regularization=1e-3,
    )

    left_indices = select_top_n_indices(
            Diff_position_left[10], N=number, by_abs=True, component=2
        )
    right_indices = select_top_n_indices(
            Diff_position_right[10], N=number, by_abs=True, component=2
        )
    # Fordis_left = Fordis_left[:,left_indices]
    # Fordis_right = Fordis_right[:,right_indices]
    ##-----------------------------------------由于参考系偏移造成的力矩的变化量-----------------------------##
    ##以初始位置作为参考
    top_points_left_init = np.array(
        [
            select_top_n(Diff_position_left[t], Position_left[t], N=number, by_abs=True)
            for t in range(10)
        ]
    ).reshape(-1, number, 3)
    top_points_right_init = np.array(
        [
            select_top_n(Diff_position_right[t], Position_right[t], N=number, by_abs=True)
            for t in range(10)
        ]
    ).reshape(-1, number, 3)
    pose_left_init = np.mean(np.mean(top_points_left_init, axis=1), axis=0)
    pose_right_init = np.mean(np.mean(top_points_right_init, axis=1), axis=0)
    pose_left_init_W = (rotation_left @ pose_left_init.T).T
    pose_right_init_W = (rotation_right @ pose_right_init.T).T  ##(3)
    ##计算任意时刻的位置
    top_points_left = np.array(
        [
            select_top_n(Diff_position_left[t], Position_left[t], N=number, by_abs=True)
            for t in range(len(Diff_position_left))
        ]
    ).reshape(-1, number, 3)
    top_points_right = np.array(
        [
            select_top_n(Diff_position_right[t], Position_right[t], N=number, by_abs=True)
            for t in range(len(Diff_position_right))
        ]
    ).reshape(-1, number, 3)
    pose_left_mean = np.mean(top_points_left, axis=1)
    pose_right_mean = np.mean(top_points_right, axis=1)  ## （T，3）
    pose_left_mean_W = np.array(
        [(rotation_left @ pose_left_mean[t].T).T for t in range(len(pose_left_mean))]
    )
    pose_right_mean_W = np.array(
        [(rotation_right @ pose_right_mean[t].T).T for t in range(len(pose_right_mean))]
    )  ##（T，3）
    ##计算相对偏移
    delta_pose_left = pose_left_mean_W - pose_left_init_W
    delta_pose_right = pose_right_mean_W - pose_right_init_W
    delta_pose = (delta_pose_left + delta_pose_right) / 2  ##(T,3)
    # #---------第一种方法
    # pose_left_mean_W = pose_left_init_W + delta_pose
    # pose_right_mean_W = pose_right_init_W + delta_pose  ##(T,3)
    # d_right = -(pose_right_mean_W + np.array([0.0, 4.0, 0.0]))
    # d_left = -(pose_left_mean_W + np.array([0.0, -4.0, 0.0]))
    ##---------第二种方法
    d_right = -(rotation_center_world_right + np.array([0.0, 4.0, 0.0]))
    d_left = -(rotation_center_world_left + np.array([0.0, -4.0, 0.0]))
    ##计算转换到世界坐标系下的力矩
    Fordis_left_W = np.array(
        [(rotation_left @ Fordis_left[t].T).T for t in range(len(Fordis_left))]
    )
    Fordis_right_W = np.array(
        [(rotation_right @ Fordis_right[t].T).T for t in range(len(Fordis_right))]
    )
    torque_left_add_world = np.array(
        [np.cross(d_left[t], Fordis_left_W[t]) for t in range(len(Fordis_left_W))]
    )  # (T, 400, 3)
    torque_right_add_world = np.array(
        [np.cross(d_right[t], Fordis_right_W[t]) for t in range(len(Fordis_right_W))]
    )  # (T, 400, 3)
    torque_additional = np.sum(torque_left_add_world, axis=1) + np.sum(
        torque_right_add_world, axis=1
    )  # (T, 3)
    ##----------------------解析力矩计算--------------------------------------------------------------------##
    rotation_center_world = estimate_rotation_center_series(
        Position_left,
        Position_right,
        Diff_position_left,
        Diff_position_right,
        rotation_left,
        rotation_right,
        dis_left=(-pose_left_init_W) + np.array([0.0, 4.0, 0.0]),
        dis_right=(-pose_right_init_W) + np.array([0.0, -4.0, 0.0]),
        top_n=number,
        min_rotation_deg=0.5,
        min_motion_mm=0.02,
        regularization=1e-3,
    )

    ## 这里使用力分布求和得到的合力数据
    F_dis_resultant = np.sum(Fordis_left_W, axis=1) + np.sum(Fordis_right_W, axis=1) #(T,3)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(F_dis_resultant[:,0])
    plt.title("F_dis_res_x")
    plt.subplot(3, 1, 2)
    plt.plot(F_dis_resultant[:,1])
    plt.title("F_dis_res_y")
    plt.subplot(3, 1, 3)
    plt.plot(F_dis_resultant[:,2])
    plt.title("F_dis_res_z")
    plt.show()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(rotation_center_world[:, 0])
    plt.title("estimated rotation center x")
    plt.subplot(3, 1, 2)
    plt.plot(rotation_center_world[:, 1])
    plt.title("estimated rotation center y")
    plt.subplot(3, 1, 3)
    plt.plot(rotation_center_world[:, 2])
    plt.title("estimated rotation center z")
    plt.tight_layout()
    plt.show()

    Rotation = np.array(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    )  ##受力点在夹爪x轴正向时的旋转矩阵。
    transtion = np.array(
        [30.0, -25.0, -117.21]
    )  ##受力点在夹爪x轴正向时的平移向量，单位mm(这里需要进一步校准)
    # Rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])  ##受力点在夹爪x轴负向时的旋转矩阵
    # transtion = np.array(
    #     [-30.0, 25.0, -128.75 - (-10)]
    # )  ##受力点在夹爪x轴负向时的平移向量，单位mm
    # Rotation = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])  ##受力点在夹爪y轴正向时的旋转矩阵
    # transtion = np.array(
    #     [25.0, 30.0, -128.75 - (-10)]
    # )  ##受力点在夹爪y轴正向时的平移向量，单位mm
    # Rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  ##受力点在夹爪y轴负向时的旋转矩阵
    # transtion = np.array(
    #     [-25.0, -30.0, -128.75 - (-10)]
    # )  ##受力点在夹爪y轴负向时的平移向量，单位mm

    torque_computed = np.zeros(0)
    contact_pose_T = np.zeros(0)
    contact_pose_T_R = np.zeros(0)
    contact_pose_T_L = np.zeros(0)
    left_initial_world = (rotation_left @ Position_left[0].T).T  + np.array([0.0, 4.0, 0.0])
    right_initial_world = (rotation_right @ Position_right[0].T).T + np.array([0.0, -4.0, 0.0])
    R_all, T_all = [], []

    for t in range(len(Position_left)):
        contact_pose = (Rotation @ np.array([0.0, 50.0, 0.0]).T).T + transtion
        sensor_to_tool_permant = np.array(
            [
                [1, 0, 0, contact_pose[0]],
                [0, 1, 0, contact_pose[1]],
                [0, 0, 1, contact_pose[2]],
                [0, 0, 0, 1],
            ]
        )  # 工具坐标系相对于传感器坐标系的转换先验（定值，需事先标定）

        # left_indices = select_top_n_indices(
        #     Diff_position_left[t], N=number, by_abs=True, component=2
        # )
        # right_indices = select_top_n_indices(
        #     Diff_position_right[t], N=number, by_abs=True, component=2
        # )
        left_current_world = (rotation_left @ Position_left[t].T).T + np.array([0.0, 4.0, 0.0])
        right_current_world = (rotation_right @ Position_right[t].T).T + np.array([0.0, -4.0, 0.0])
        left_initial_active = left_initial_world[left_indices]
        left_current_active = left_current_world[left_indices]
        right_initial_active = right_initial_world[right_indices]
        right_current_active = right_current_world[right_indices]

        # initial_active = np.vstack((left_initial_active, right_initial_active))
        # current_active = np.vstack((left_current_active, right_current_active))
        # R, T = estimate_rigid_transform_kabsch(
        #     initial_active, current_active
        # )
        # sensor_sense_translation = np.array(
        #     [
        #         [R[0, 0], R[0, 1], R[0, 2], T[0]],
        #         [R[1, 0], R[1, 1], R[1, 2], T[1]],
        #         [R[2, 0], R[2, 1], R[2, 2], T[2]],
        #         [0, 0, 0, 1],
        #     ]
        # )  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
        # sensor_to_tool_translation = (
        #         sensor_sense_translation @ sensor_to_tool_permant
        # )
        # contact_cuur = (sensor_to_tool_translation @ np.array([0, 0, 0, 1]).T)[0:3]
        # contact_pose_T = np.append(contact_pose_T, contact_cuur).reshape(-1, 3)

        R, T = estimate_rigid_transform_kabsch(
            left_initial_active, left_current_active
        )

        sensor_sense_translation_left = np.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], T[0]],
                [R[1, 0], R[1, 1], R[1, 2], T[1]],
                [R[2, 0], R[2, 1], R[2, 2], T[2]],
                [0, 0, 0, 1],
            ]
        )  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
        sensor_to_tool_translation_left = (
            sensor_sense_translation_left @ sensor_to_tool_permant
        )
        contact_cuur_l = (sensor_to_tool_translation_left @ np.array([0, 0, 0, 1]).T)[0:3]
        contact_pose_T_L = np.append(contact_pose_T_L, contact_cuur_l).reshape(-1, 3)

        R, T = estimate_rigid_transform_kabsch(
            right_initial_active, right_current_active
        )

        sensor_sense_translation_right = np.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], T[0]],
                [R[1, 0], R[1, 1], R[1, 2], T[1]],
                [R[2, 0], R[2, 1], R[2, 2], T[2]],
                [0, 0, 0, 1],
            ]
        )  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
        sensor_to_tool_translation_right = (
            sensor_sense_translation_right @ sensor_to_tool_permant
        )
        contact_cuur_r = (sensor_to_tool_translation_right @ np.array([0, 0, 0, 1]).T)[0:3]
        contact_pose_T_R = np.append(contact_pose_T_R, contact_cuur_r).reshape(-1, 3)

        contact_cuur = (contact_cuur_l + contact_cuur_r) / 2
        contact_pose_T = np.append(contact_pose_T, contact_cuur).reshape(-1, 3)

        # 计算合力向量
        F = np.column_stack([F_x[t], F_y[t], F_z[t]]).reshape(3)  # 形状 (3)
        # F = F_dis_resultant[t]
        # 计算合力矩：torque = contact_pose × F
        torque_computed = np.append(torque_computed, np.cross(contact_cuur, F)+np.cross(-rotation_center_world[t], F)).reshape(
            -1, 3
        )  # 形状 (T, 3)

        R_all.append(R)
        T_all.append(T)

    # 提取合力矩分量
    torque_x = torque_computed[:, 0]
    torque_y = torque_computed[:, 1]
    torque_z = torque_computed[:, 2]

    ###---------------根据传感器得到的力分布计算力矩-------------------------------------------##
    # rotation_left = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    # rotation_right = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    # 计算传感器坐标系下的力矩
    torque_left_sensor = np.cross(Position_left, Fordis_left)  # (T, 400, 3)
    torque_right_sensor = np.cross(Position_right, Fordis_right)  # (T, 400, 3)

    # 转化到世界坐标系
    torque_left_world = np.array(
        [
            (rotation_left @ torque_left_sensor[t].T).T
            for t in range(len(torque_left_sensor))
        ]
    )
    torque_right_world = np.array(
        [
            (rotation_right @ torque_right_sensor[t].T).T
            for t in range(len(torque_right_sensor))
        ]
    )

    # 计算总力矩
    total_torque_left = np.sum(torque_left_world, axis=1)  # (T, 3)
    total_torque_right = np.sum(torque_right_world, axis=1)  # (T, 3)
    # total_torque = total_torque_left + total_torque_right
    total_torque_all = total_torque_left + total_torque_right + torque_additional
    # total_torque_rotation_center = total_torque_all - np.cross(
    #     rotation_center_world, F_dis_resultant
    # )

    # 画图
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(total_torque_all[:, 0] - total_torque_all[:, 0][0], label="estimated Torque x")
    plt.plot(torque_x - torque_x[0], label="computed Torque_x")
    plt.title("X")
    plt.subplot(3, 1, 2)
    plt.plot(total_torque_all[:, 1] - total_torque_all[:, 1][0], label="estimated Torque y")
    plt.plot(torque_y - torque_y[0], label="computed Torque_y")
    plt.title("Y")
    plt.subplot(3, 1, 3)
    plt.plot(total_torque_all[:, 2] - total_torque_all[:, 2][0], label="estimated Torque z")
    plt.plot(torque_z - torque_z[0], label="computed Torque_z")
    plt.title("Z")
    plt.tight_layout()
    plt.legend()
    plt.show()

    ##----------------------绘制传感器点图----------------------------------##
    xs, ys, zs = zip(*left_initial_world)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c='g', marker='^')
    xr, yr, zr = zip(*left_initial_world[left_indices])
    ax.scatter(xr, yr, zr, c='r', s=60)
    # 高亮 top N 点
    # xs, ys, zs = zip(*right_initial_world)
    # sc = ax.scatter(xs, ys, zs, c='g', marker='^')
    # xr, yr, zr = zip(*right_initial_world[right_indices])
    # ax.scatter(xr, yr, zr, c='r', s=60)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Positions')
    # ax.legend()
    plt.show()

