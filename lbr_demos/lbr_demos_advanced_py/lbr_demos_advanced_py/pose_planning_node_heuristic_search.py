import math
import pickle

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Vector3
import numpy as np
from .lazy_rtdp_bel_ablation_test import Motion_planning
# from .motion_planning_2 import Motion_planning
from tutorial_interfaces.msg import Cloud, Array3
from .tac3d_test import gripper_force_direct
from .resources import package_resource

"""
socket assembly的正式版本。(还待完善运动趋势估计)
"""


class MinimumJerkPosePlanner(Node):
    def __init__(self):
        super().__init__('min_jerk_pose_planner_py')
        self.v_move = 0.0
        self.N_ite = None
        self.mu = None
        self.dz_flag = None
        self.dhori = None
        self.Q = None
        self.cmd = Pose()
        self.v_vec = None
        self.initial = None
        self.current = None
        self.target = None
        self.total_time = 0.0
        self.sampling_time = 0.01
        self.freq = 100
        self.start_time = None
        self.X, self.Y, self.Z = None, None, None

        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 1)
        self.pose_sub = self.create_subscription(Pose, '/lbr/state/pose',
                                                 self.on_pose, 1)

        self.F_l = None  # 左手力分布
        self.F_r = None  # 右手力分布
        self.Fr_l = None  # 左手合力数据
        self.Fr_r = None  # 右手合力数据
        self.Mr_l = None  # 左手合力数据
        self.Mr_r = None  # 右手合力数据

        self.Force_left = []  # 左手合力集合
        self.Force_right = []  # 右手合力集合
        self.Matrix_left = []
        self.Matrix_right = []
        self.Force_dis_left = []  # 左手力分布集合
        self.Force_dis_right = []  # 右手力分布集合
        self.F_x, self.F_y, self.F_z, self.R_x, self.R_y, self.R_z = [], [], [], [], [], []
        self.Pose_all, self.Time = [], []

        self.force_inside_flag = False
        self.force_control_flag = False
        self.move_hori_flag = False
        self.i = 0
        self.N = 0
        self.n_now = 0
        self.terminate_flag = False
        self.terminate_num = 0.0

        self.window_size = 6
        self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.count = 0
        self.pos = 0

        self.m = 0
        self.F_x_1 = np.zeros(0)
        self.F_y_1 = np.zeros(0)
        self.F_z_1 = np.zeros(0)
        self.f_x_mean = None
        self.f_y_mean = None
        self.f_z_mean = None

        ##进行主动探索所需要的参数
        self.peaks = 0  ##峰值数目
        self.max_points = []  ##权重符合条件的粒子位置和权重
        pkl_path = package_resource("socket_contours_2.pkl")
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        self.obstacles_all = loaded
        R = np.array([[0, -1], [1, 0]])  # 顺时针90°旋转矩阵
        # rotated = []
        # for c in self.obstacles_all:
        #     rotated.append(c @ R.T)  # 矩阵乘法
        # self.obstacles_all = rotated
        ## 二脚插头
        # self.Movable_objects = [
        #      np.array([[-7, -3.2], [-5.5, -3.2], [-5.5, 3.2], [-7, 3.2]], dtype=float),  # 小矩形1
        #      np.array([[5.5, -3.2], [7, -3.2], [7, 3.2], [5.5, 3.2]], dtype=float)]  # 小矩形2（与1相对位置固定）
        ## 三脚插头(注意参考点变了，则坐标也要相应改变)
        # self.Movable_objects = [np.array([[-0.75, 5.98-12.38], [0.75, 5.98-12.38], [0.75, 0.0], [-0.75, 0.0]])]
        ## 二指圆孔插销
        self.Movable_objects = [
            np.array(
                [
                    [11.8, 0.0],
                    [11.126346, 1.626346],
                    [9.5, 2.3],
                    [7.873654, 1.626346],
                    [7.2, 0.0],
                    [7.873654, -1.626346],
                    [9.5, -2.3],
                    [11.126346, -1.626346],
                ],
                dtype=float,
            ),  # 小多边形1
            np.array(
                [
                    [-7.2, 0.0],
                    [-7.873654, 1.626346],
                    [-9.5, 2.3],
                    [-11.126346, 1.626346],
                    [-11.8, 0.0],
                    [-11.126346, -1.626346],
                    [-9.5, -2.3],
                    [-7.873654, -1.626346],
                ],
                dtype=float,
            ),  # 小多边形2（与1相对位置固定）
        ]

        rotated = []
        for c in self.Movable_objects:
            rotated.append(c @ R.T)  # 矩阵乘法
        self.Movable_objects = rotated
        self.particles = None
        self.particles_filter = None
        self.weights_filter = None
        self.z = None
        self.weights = None
        self.Z_obs = []
        self.Z_obs_pre = []
        self.num = 0  ##这是一个计算机器人进入几次力调整的次数的量
        self.Pose = np.zeros(0)
        self.Pose_pre = np.zeros(0)
        self.v_pred = np.zeros(0)
        self.v_pred_pre = np.zeros(0)
        self.F_x_N, self.F_y_N, self.F_z_N = [], [], []
        ## 二指插座的间距设置
        # curve_dis = np.array([9.4, 34.3, 14.75, 14.75])  ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
        ## 三指插座的间距设置(在更改了超出距离判定之后的的间距设置)
        # curve_dis = np.array([26.35, 31.11, 16.5, 16.5])  ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
        ## 二指圆孔插座间距设置
        curve_dis = np.array(
            [8.6, 33.5, 19.55, 19.55]
        )  ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
        self.M = Motion_planning(d=0.25, padding=curve_dis)  # 这里的单位是mm
        self.scene, self.xs, self.ys = self.M.discretize_scene(self.obstacles_all)

        self.P_l_sub = self.create_subscription(
            Cloud, '/positions_l', self.on_P_l, 10)
        self.P_r_sub = self.create_subscription(
            Cloud, '/positions_r', self.on_P_r, 10)
        self.D_l_sub = self.create_subscription(
            Cloud, '/displacements_l', self.on_D_l, 10)
        self.D_r_sub = self.create_subscription(
            Cloud, '/displacements_r', self.on_D_r, 10)
        self.F_l_sub = self.create_subscription(
            Cloud, '/forces_l', self.on_F_l, 10)
        self.F_r_sub = self.create_subscription(
            Cloud, '/forces_r', self.on_F_r, 10)
        self.Fr_l_sub = self.create_subscription(
            Vector3, '/resultant_force_l', self.on_Fr_l, 10)
        self.Fr_r_sub = self.create_subscription(
            Vector3, '/resultant_force_r', self.on_Fr_r, 10)
        self.Mr_l_sub = self.create_subscription(
            Vector3, '/resultant_moment_l', self.on_Mr_l, 10)
        self.Mr_r_sub = self.create_subscription(
            Vector3, '/resultant_moment_r', self.on_Mr_r, 10)

        # # 可选：从话题接收目标 Pose
        # self.target_sub = self.create_subscription(Pose, 'target/pose',
        #                                            self.on_target, 1

    def on_P_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.P_l = xyz
        # self.get_logger().info(f'P_l array shape: {xyz.shape}')

    def on_P_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.P_r = xyz
        # self.get_logger().info(f'P_r array shape: {xyz.shape}')

    def on_D_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.D_l = xyz
        # self.get_logger().info(f'D_l array shape: {xyz.shape}')

    def on_D_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.D_r = xyz
        # self.get_logger().info(f'D_r array shape: {xyz.shape}')

    def on_F_l(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_l = xyz
        self.Force_dis_left.append(self.F_l)
        # self.get_logger().info(f'F_l array shape: {xyz.shape}')

    def on_F_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_r = xyz
        self.Force_dis_right.append(self.F_r)
        # self.get_logger().info(f'F_r array shape: {xyz.shape}')

    def on_Fr_l(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_l = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Force_left.append(self.Fr_l)
            # self.get_logger().info(f'Fr_l: {self.Fr_l}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Fr_r(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_r = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Force_right.append(self.Fr_r)
            # self.get_logger().info(f'Fr_r: {self.Fr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Mr_l(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Mr_l = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Matrix_left.append(self.Mr_l)
            # self.get_logger().info(f'Mr_r: {self.Mr_l}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_Mr_r(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Mr_r = np.array([x_val, y_val, z_val]).reshape(1, 3)
            self.Matrix_right.append(self.Mr_r)
            # self.get_logger().info(f'Mr_r: {self.Mr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_pose(self, msg):
        if self.initial is None:  ##确定任务规划的起点
            self.initial = msg
            self.cmd.position.x = self.initial.position.x
            self.cmd.position.y = self.initial.position.y
            self.cmd.position.z = self.initial.position.z
            self.cmd.orientation = self.initial.orientation
            self.get_logger().info('Initial pose received.')
        else:
            # 如果已有初始与目标 且 未启动轨迹
            if self.start_time is None:
                if any(v is None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
                    self.get_logger().info('Waiting for tac3d information.')
                else:
                    if -10 < self.Fr_r[0, 2] < -8 and -10 < self.Fr_l[0, 2] < -8 and self.force_inside_flag == False:
                        # self.force_inside_flag = True
                        self.start_time = self.get_clock().now().to_msg().sec + \
                                          self.get_clock().now().to_msg().nanosec * 1e-9
                        self.get_logger().info('Starting minimum jerk trajectory.')
                    else:
                        if self.i % 100 == 0:
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
            # 如果轨迹已经启动
            if self.start_time is not None:
                if -15 < self.Fr_r[0, 2] < -5 and -15 < self.Fr_l[0, 2] < -5:
                    self.force_inside_flag = True
                else:
                    self.force_inside_flag = False
                    self.get_logger().info('The force is out of range!.')
                    self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')

                if self.force_inside_flag:
                    self.current = msg
                    if self.m < 20:
                        f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
                        self.F_x_1 = np.append(self.F_x_1, f_x)
                        self.F_y_1 = np.append(self.F_y_1, f_y)
                        self.F_z_1 = np.append(self.F_z_1, f_z)
                        self.m += 1
                    elif self.m == 20:
                        self.f_x_mean = np.mean(self.F_x_1)
                        self.f_y_mean = np.mean(self.F_y_1)
                        self.f_z_mean = np.mean(self.F_z_1)
                        self.get_logger().info(
                            f'The mean initial force is f_x={self.f_x_mean}, f_y={self.f_y_mean}, f_z={self.f_z_mean}')
                        self.m += 1
                    else:
                        self.publish_at_time()

        self.i += 1

    def clean_single_float(self, value):
        """清理单个浮点数，处理NaN和无穷大等奇异值"""
        import math

        try:
            # 转换为浮点数
            float_val = float(value)

            # 检查是否为NaN或无穷大
            if math.isnan(float_val) or math.isinf(float_val):
                return -99.0
            # 检查是否为异常大的数值（绝对值超过1e10）
            elif abs(float_val) > 1e10:
                return -99.0
            else:
                return float_val
        except (ValueError, TypeError):
            return -99.0  # 转换失败时返回-99

    def As_planner(self):
        if self.initial is not None and self.target is not None:  ##满足轨迹规划的基本条件
            start = np.array([self.initial.position.x, self.initial.position.y, self.initial.position.z])
            # end = np.array([self.target.position.x, self.target.position.y, self.target.position.z])
            # Points_recover = M.path_searching(start=start, end=end)
            self.total_time = 0.0
            Points_recover = []
            Points_recover.append(start)  ##将规划的初始点纳入
            q0 = np.array([self.initial.orientation.x,
                           self.initial.orientation.y,
                           self.initial.orientation.z,
                           self.initial.orientation.w])
            qd = q0
            self.mu = 1.0
            self.v_move = 0.003  ##移动的速度（m/s）
            if self.N == 0:  ## 初始时机械臂向下运动直到接触孔平面
                Points_recover.append(start + np.array([0.0, 0.0, -0.3]))  # 这里的单位是m
                Points_recover.append(start + np.array([0.0, 0.0, -0.6]))
                self.v_pred = np.array([0, 0])  ##末端只在竖直方向上运动，横向速度均为0
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = np.zeros(0)  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.F_x_N, self.F_y_N, self.F_z_N = [], [], []
                self.m = 0

            elif self.N == 1:  ##初始接触孔平面，需要进一步选取运动方向
                self.dz_flag = 0
                # 检测峰值，选择相应的方向
                peaks = self.M.find_local_maximum(self.particles, self.weights, radius=5)
                self.get_logger().info(f'len(peaks) = {len(peaks)}')
                self.v_pred = self.M.lazy_rtdp_bel_action(self.particles, self.weights, self.Movable_objects)  ##需进一步改进
                print(f'v_pred = {self.v_pred}')
                # ===== 新增: 等待键盘输入 'q' 才继续 =====
                self.get_logger().info("Change v_pred and Press 'q' and Enter to continue...")
                while True:
                    user_in = input()
                    if user_in == 'q':
                        break
                    elif user_in == 'w':
                        try:
                            new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                            new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                            self.v_pred = np.array([new_val_0, new_val_1])
                            self.get_logger().info(f"self.v_pred 已更新为 {self.v_pred}")
                        except ValueError:
                            self.get_logger().warn("输入无效，请输入数值。")
                    else:
                        print("无效输入，请输入 'q' 或 'w'")
                # 将运动方向转化为运动距离
                delta = self.v_pred * 0.3  ##这里的单位为m
                # 加入到轨迹规划中
                Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.0]))
                Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = np.zeros(0)  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.F_x_N, self.F_y_N, self.F_z_N = [], [], []

            elif self.N == 2:
                current_pose = np.array(
                    [self.current.position.x, self.current.position.y])
                delta = -(current_pose - self.Pose[0])  ##这里单位为m
                self.particles = self.particles + delta * 1000  ##更新粒子的位置，将粒子的分布也返回上一次出发点（单位转化成mm）
                self.get_logger().info('Move the particles to last initial position!')
                Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.04]))  ##适当向上运动
                Points_recover.append(start + np.array([delta[0], delta[1], 0.04]))
                self.v_pred = np.array([0, 0])  ##这里是回归上一时刻的出发位置。
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = np.zeros(0)  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.F_x_N, self.F_y_N, self.F_z_N = [], [], []

            elif self.N == 3:  ##在还未收敛时落入孔区域，检测是否对齐
                if self.dz_flag == 1:  ##运动到目标位置上方，向下运动接触孔平面
                    Points_recover.append(start + np.array([0.0, 0.0, -0.1]))  # 这里的单位是m
                    Points_recover.append(start + np.array([0.0, 0.0, -0.2]))
                    self.get_logger().info('-------Try insert!!!-----------')
                else:
                    self.get_logger().info(f"The current dz_flag={self.dz_flag}, The error happens!")

                # ===== 新增: 等待键盘输入 'q' 才继续 =====
                self.get_logger().info("The current state is N=3.")
                self.get_logger().info(f"The current dz_flag={self.dz_flag}")
                while True:
                    user_in = input()
                    if user_in == 'q':
                        break
                    else:
                        print("无效输入，请输入 'q' 或 'w'")

                self.z = 0  ## 触觉信号置0,重新开始观测
                self.v_pred = np.array([0, 0])  ##末端只在竖直方向上运动，横向速度均为0
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = np.zeros(0)  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.F_x_N, self.F_y_N, self.F_z_N = [], [], []

            elif self.N == 4:  ##在探索收敛后，进行
                hole_pose = np.array([-45.83 - 6.25, 23.55])  ##三脚插座的位置
                hole_state = [1, 1, 1, 0]  ##三脚插座的目标接触状态依次（[y+,y-,x+,x-]）
                if self.dz_flag == 0:  ##粒子分布收敛，准备运动到孔附近上方
                    ## 计算粒子当前在插销坐标系下的位置
                    particles_pose = np.mean(self.particles_filter, axis=0)
                    self.particles = self.particles_filter  ##粒子只保留权重高的部分
                    ## 计算粒子的当前位置与目标位置之间的差值
                    ## 运动
                    target_pose = hole_pose + np.array([0.0, -3.0])  ##粒子先运动到孔位置一侧的上方
                    delta = -(particles_pose - target_pose) / 1000  ##这里的单位为m
                    self.particles = self.particles + delta * 1000  ##更新粒子的位置(单位为mm)
                    self.get_logger().info(f"The current pin pose: {particles_pose}.")
                    self.get_logger().info('Move the particles to ahead of hole!')
                    Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.03]))  ##适当向上运动
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.03]))
                    self.v_pred = np.array([0, 0])  ##末端只在竖直方向上运动，横向速度均为0
                    self.N_ite = 0
                if self.dz_flag == 1:  ##已运动到目标位置上方，向下运动接触孔平面
                    Points_recover.append(start + np.array([0.0, 0.0, -0.1]))  # 这里的单位是m
                    Points_recover.append(start + np.array([0.0, 0.0, -0.2]))
                    self.get_logger().info('-------Contact!!!-----------')
                    self.v_pred = np.array([0, 0])  ##末端只在竖直方向上运动，横向速度均为0
                if self.dz_flag == 2:  ## 接触孔平面后，，向孔方向运动，同时检测是否符合约束条件(约束条件就是Z方向无力，横向有约束)
                    V_list = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
                    self.v_pred = V_list[self.N_ite]
                    self.get_logger().info('choose the current velocity direction based on Y+Y-X-X+')
                    self.get_logger().info(f"The chosen velocity is:{self.v_pred}")
                    while True:
                        user_in = input()
                        if user_in == 'q':
                            break
                        elif user_in == 'w':
                            try:
                                new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                                new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                                self.v_pred = np.array([new_val_0, new_val_1])
                                self.get_logger().info(f"self.v_pred 已更新为 {self.v_pred}")
                            except ValueError:
                                self.get_logger().warn("输入无效，请输入数值。")
                        else:
                            print("无效输入，请输入 'q' 或 'w'")

                    # 将运动方向转化为运动距离
                    delta = self.v_pred * 0.008 * self.N_ite  ##这里的单位为m
                    # 加入到轨迹规划中
                    Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info('-------To Find Constrain!!!-----------')
                    self.N_ite += 1
                    self.v_move = 0.0006  ##此时的移动速度降为原速度的五分之一

                self.get_logger().info("The current state is N=4.")
                self.get_logger().info(f"The current dz_flag={self.dz_flag}")

                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = np.zeros(0)  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.F_x_N, self.F_y_N, self.F_z_N = [], [], []

            elif self.N > 4:
                self.get_logger().info('End the trajectory!')
            ## 姿态的规划
            if (self.v_pred == np.array([0, 1])).all():
                qd = np.array([0.0,
                               math.cos(-5 * math.pi / 180),
                               math.sin(-5 * math.pi / 180),
                               0.0])
                # qd = np.array([0.0,
                #                1.0,
                #                0.0,
                #                0.0])
                self.get_logger().info('move along Y+!')
                self.mu = 0.58
                # self.mu = 0.47
            elif (self.v_pred == np.array([0, -1])).all():
                qd = np.array([0.0,
                               math.cos(5 * math.pi / 180),
                               math.sin(5 * math.pi / 180),
                               0.0])
                # qd = np.array([0.0,
                #                1.0,
                #                0.0,
                #                0.0])
                self.get_logger().info('move along Y-!')
                self.mu = 0.58
            elif (self.v_pred == np.array([1, 0])).all():
                qd = np.array([0.0,
                               1.0,
                               0.0,
                               5.0 * math.pi / 180])
                # qd = np.array([0.0,
                #                1.0,
                #                0.0,
                #                0.0])
                self.get_logger().info('move along X+!')
                # self.mu = 0.17
                self.mu = 0.47
            elif (self.v_pred == np.array([-1, 0])).all():
                qd = np.array([0.0,
                               1.0,
                               0.0,
                               -5.0 * math.pi / 180])
                # qd = np.array([0.0,
                #                1.0,
                #                0.0,
                #                0.0])
                self.get_logger().info('move along X-!')
                self.mu = 0.47
            else:
                qd = np.array([0.0,
                               1.0,
                               0.0,
                               0.0])
                self.get_logger().info('move along other direction!')
                self.mu = 1.0

            for j in range(len(Points_recover) - 1):
                self.total_time += max(np.linalg.norm(Points_recover[j + 1] - Points_recover[j]) / self.v_move,
                                       abs(np.arccos(np.clip(np.dot(q0, qd), -1.0, 1.0)) * 2.0) / 0.03)

            if Points_recover is not None:
                print(f'current_pose = {self.current}')
                # ===== 新增: 等待键盘输入 'q' 才继续 =====
                self.get_logger().info("Press 'q' and Enter to continue...")
                while True:
                    user_in = input()
                    if user_in == 'q':
                        break
                    else:
                        print("无效输入，请输入 'q' 或 'w'")

                self.X, self.Y, self.Z, self.Q = self.M.path_smoothing(Path_points=Points_recover, q0=q0, qf=qd,
                                                                       t_final=self.total_time,
                                                                       freq=self.freq)  ##轨迹使用二次B样条曲线进行平滑处理
                self.get_logger().info("path_points get !")
            else:
                self.get_logger().info("the path is not found!!")

    def publish_at_time(self):
        self.cmd = self.current  ##发布量的原始值为当前位置
        current_time = self.get_clock().now().to_msg().sec + \
                       self.get_clock().now().to_msg().nanosec * 1e-9
        ##进行触觉反馈探索
        if all(v is not None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
            f_x = f_x - self.f_x_mean
            f_y = f_y - self.f_y_mean
            f_z = f_z - self.f_z_mean
            self.F_x.append(f_x)
            self.F_y.append(f_y)
            self.F_z.append(f_z)
            self.R_x.append(r_x), self.R_y.append(r_y), self.R_z.append(r_z)
            self.Pose_all.append(np.array(
                [self.current.position.x, self.current.position.y, self.current.position.z]))
            self.Time.append(current_time)

            ##计算合力矢量
            f_vec = np.array([f_x, f_y, f_z])

            ##计算偏移量(刚度越大，偏移量应该越小)
            delta_z = np.sign(f_z - 2.1) * min(0.0032, 0.0032 * abs(f_z - 2.1)) if abs(f_z) > 0.01 else 0.0
            delta_x = np.sign(f_x - 0.05) * min(0.005, 0.005 * abs(f_x - 0.05)) if abs(f_x) > 0.1 else 0.0
            delta_y = np.sign(f_y - 0.1) * min(0.005, 0.005 * abs(f_y - 0.1)) if abs(f_y) > 0.2 else 0.0

            if self.force_control_flag:  ##接触力调整与推理阶段，保持一个恒定的接触力
                self.cmd.position.x += delta_x / self.freq
                self.cmd.position.y += delta_y / self.freq
                self.cmd.position.z += delta_z / self.freq

                if self.N == 4:
                    if self.dz_flag == 1:
                        if f_z > 1.0:
                            self.dz_flag = 2
                        else:
                            self.get_logger().info('Don\'t contact with surface')
                    elif self.dz_flag == 2:
                        if self.F_z_N[-1] < 0.9:  ## 小于
                            self.get_logger().info('Have inside the hole range')
                            self.dz_flag = 1  ## 继续向下运动使得完成插入过程
                        else:
                            self.get_logger().info('Haven\'t find the hole range')

                    ##保存相关数据进行分析
                    np.savez(f'contact_slam_N4data_2511101625_{str(self.num)}.npz', particles=self.particles,
                             weights=self.weights, Pose_list=self.Pose_pre,
                             particles_filter=self.particles_filter,
                             Z_list=self.Z_obs_pre, Time=self.Time, v_pred=self.v_pred_pre, scene=self.scene,
                             xs=self.xs, ys=self.ys)
                    self.get_logger().info(f'The N4 data has been saved!')

                    self.force_control_flag = False
                    self.initial = None  ##此时再接收初始位置
                    self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
                    self.sum = np.zeros(3, dtype=np.float64)
                    self.count = 0
                    self.pos = 0  ##这些计算平均力的值也要归零
                    self.num += 1  ##这是一个计算机器人进入几次力调整的标志量
                    self.X, self.Y, self.Z = None, None, None

                if self.N == 3:  ## 还未收敛时落入孔内判断是否与孔对齐
                    d_pose = np.zeros(0)
                    for i in range(len(self.F_x_N)):
                        if self.F_z_N[i] > 0.2:
                            d_pose = self.Pose[-1] - self.Pose[i]  ##计算竖直方向从受力到超出阈值的纵向运动距离
                            break
                    self.get_logger().info(f'The vertical distance of insertion is: {d_pose}m')

                    d_height = np.linalg.norm(d_pose)
                    if d_height > 0.02:  ##具体数值可再修改
                        self.get_logger().info(f'The insertion has been finished!')
                    else:
                        self.get_logger().info(f'The insertion has failed!')
                        self.dz_flag = 0
                        self.N = 1
                        ## 插入失败后，进行正常的粒子滤波迭代
                        particle_range = self.M.main_line_perception(self.Z_obs_pre,
                                                                     self.v_pred_pre, self.Movable_objects)  ##这里的速度有问题
                        self.particles, self.weights, particles_remain = self.M.update_particles(self.particles,
                                                                                                 self.weights,
                                                                                                 particle_range,
                                                                                                 self.Z_obs_pre,
                                                                                                 shift_3=(self.Pose_pre[
                                                                                                              -1] -
                                                                                                          self.Pose_pre[
                                                                                                              0]) * 1000)
                        # 更新权重
                        self.weights = self.M.weights_updates_dis(action=self.v_pred_pre,
                                                                  Movable_objects_1=self.Movable_objects,
                                                                  z_obs_list=self.Z_obs_pre,
                                                                  Pos_list_1=self.Pose_pre * 1000,
                                                                  particles_5=self.particles,
                                                                  weights_5=self.weights)

                        # 更新粒子，筛掉低权重粒子
                        w_thresh = 1.0 / (len(self.particles) * 2)
                        mask = self.weights >= w_thresh
                        self.particles_filter = self.particles[mask]
                        self.weights_filter = self.weights[mask]
                        std = np.std(self.particles_filter, axis=0)
                        if std[0] < 0.8 and std[1] < 0.8:
                            self.N = 4  ## 当粒子的范围足够小时，进入装配阶段
                            self.get_logger().info('The position is accurate enough!')
                            self.get_logger().info(f'The current particles distribution is {self.particles_filter}')
                        else:
                            self.particles, self.weights = self.M.resample_particles(self.particles_filter,
                                                                                     self.weights_filter,
                                                                                     particles_remain)

                        self.get_logger().info('The weights has update!!')
                        self.get_logger().info(f'len(particles)={len(self.particles)}')
                        self.get_logger().info(f'contact state:N={self.N}')

                    ##保存相关数据进行分析
                    np.savez(f'contact_slam_N3data_2511101625_{str(self.num)}.npz', particles=self.particles,
                             weights=self.weights, Pose_list=self.Pose_pre, particles_filter=self.particles_filter,
                             Z_list=self.Z_obs_pre, Time=self.Time, v_pred=self.v_pred_pre, scene=self.scene,
                             xs=self.xs, ys=self.ys)
                    self.get_logger().info(f'The N3 data has been saved!')

                    self.force_control_flag = False
                    self.initial = None  ##此时再接收初始位置
                    self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
                    self.sum = np.zeros(3, dtype=np.float64)
                    self.count = 0
                    self.pos = 0  ##这些计算平均力的值也要归零
                    self.num += 1  ##这是一个计算机器人进入几次力调整的标志量

                ##对于超出运动范围的情况，
                if self.N == 2:
                    particle_range = self.M.main_line_perception(self.Z_obs,
                                                                 self.v_pred, self.Movable_objects)
                    # if self.num == 1:
                    #     self.particles, self.weights = self.M.generate_particles_in_polygon(particle_range, N=2000)
                    # else:
                    #     self.particles, self.weights = self.M.update_particles(self.particles, self.weights,
                    #                                                            particle_range,
                    #                                                            shift_3=(self.Pose[-1] - self.Pose[
                    #                                                                0]) * 1000)
                    self.particles, self.weights, particles_remain = self.M.update_particles(self.particles,
                                                                                             self.weights,
                                                                                             particle_range, self.Z_obs,
                                                                                             shift_3=(self.Pose[-1] -
                                                                                                      self.Pose[
                                                                                                          0]) * 1000)

                    # 更新权重
                    self.weights = self.M.weights_updates_dis(action=self.v_pred,
                                                              Movable_objects_1=self.Movable_objects,
                                                              z_obs_list=self.Z_obs,
                                                              Pos_list_1=self.Pose * 1000, particles_5=self.particles,
                                                              weights_5=self.weights)

                    # 更新粒子，筛掉低权重粒子
                    w_thresh = 1.0 / (len(self.particles) * 2)
                    mask = self.weights >= w_thresh
                    self.particles_filter = self.particles[mask]
                    self.weights_filter = self.weights[mask]
                    std = np.std(self.particles_filter, axis=0)
                    if std[0] < 0.8 and std[1] < 0.8:
                        self.N = 4  ## 当粒子的范围足够小时，进入装配阶段
                        self.get_logger().info('The position is accurate enough!')
                        self.get_logger().info(f'The current particles distribution is {self.particles_filter}')
                    else:
                        self.particles, self.weights = self.M.resample_particles(self.particles_filter,
                                                                                 self.weights_filter,
                                                                                 particles_remain)

                    self.get_logger().info('The weights has update!!')
                    self.get_logger().info(f'len(particles)={len(self.particles)}')
                    self.get_logger().info(f'contact state:N={self.N}')
                    ##保存相关数据进行分析
                    np.savez(f'contact_slam_N2data_2511101625_{str(self.num)}.npz', particles=self.particles,
                             weights=self.weights, Pose_list=self.Pose, particles_filter=self.particles_filter,
                             Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                             xs=self.xs, ys=self.ys)
                    self.get_logger().info(f'The N2 data has been saved!')

                    self.force_control_flag = False
                    self.initial = None  ##此时再接收初始位置
                    self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
                    self.sum = np.zeros(3, dtype=np.float64)
                    self.count = 0
                    self.pos = 0  ##这些计算平均力的值也要归零
                    self.num += 1  ##这是一个计算机器人进入几次力调整的标志量

                ##合力在运动方向上的分量
                f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                if self.count < self.window_size:  ##计算最近若干个值的均值
                    self.sum += f_projec
                    self.buffer[self.pos] = f_projec
                    self.count += 1
                    self.pos = (self.pos + 1) % self.window_size
                else:
                    self.sum += f_projec - self.buffer[self.pos]
                    self.buffer[self.pos] = f_projec
                    self.pos = (self.pos + 1) % self.window_size
                    f_projec_mean = self.sum / self.count
                    if self.mu == 1.0:
                        f_ref = 2.0
                    else:
                        f_ref = f_z

                    if self.mu * f_ref * 0.9 < np.linalg.norm(
                            f_projec_mean) < self.mu * f_ref * 1.1:  ##分量的大小最近若干个值均在范围内，则认为接触力调整完成。
                        self.force_control_flag = False
                        self.get_logger().info('Contact force within threshold!')
                        ##进入粒子权重更新阶段
                        if self.N == 0:
                            if self.num == 0:
                                # 初始化粒子分布
                                particle_range = np.array([[self.xs[0], self.ys[0]], [self.xs[-1], self.ys[0]],
                                                           [self.xs[-1], self.ys[-1]], [self.xs[0], self.ys[-1]]])
                                self.particles, self.weights = self.M.generate_particles_in_polygon([particle_range],
                                                                                                    N=2000)

                            self.N = 1
                            self.get_logger().info('Transfer to exploration phase!')
                            self.get_logger().info(f'len(particles)={len(self.particles)}')
                            self.get_logger().info(f'contact state:N={self.N}')
                            # self.get_logger().info(f'range:x_min={self.M.x_min},x_max={self.M.x_max}')
                            ##保存相关数据进行分析
                            np.savez(f'contact_slam_N0data_2511101625_{str(self.num)}.npz', particles=self.particles,
                                     weights=self.weights, Pose_list=self.Pose,
                                     Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                                     xs=self.xs, ys=self.ys)
                            self.get_logger().info(f'The N0 data has been saved!')
                            self.num += 1  ##这是一个计算机器人进入几次力调整的标志量

                        elif self.N == 1:
                            particle_range = self.M.main_line_perception(self.Z_obs,
                                                                         self.v_pred, self.Movable_objects)
                            # if self.num == 1:
                            #     self.particles, self.weights = self.M.generate_particles_in_polygon(particle_range,
                            #                                                                         N=2000)
                            # else:
                            #     self.particles, self.weights = self.M.update_particles(self.particles, self.weights,
                            #                                                            particle_range,
                            #                                                            shift_3=(self.Pose[-1] -
                            #                                                                     self.Pose[0]) * 1000)
                            self.particles, self.weights, particles_remain = self.M.update_particles(self.particles,
                                                                                                     self.weights,
                                                                                                     particle_range,
                                                                                                     self.Z_obs,
                                                                                                     shift_3=(self.Pose[
                                                                                                                  -1] -
                                                                                                              self.Pose[
                                                                                                                  0]) * 1000)

                            # 更新权重
                            self.weights = self.M.weights_updates_dis(action=self.v_pred,
                                                                      Movable_objects_1=self.Movable_objects,
                                                                      z_obs_list=self.Z_obs,
                                                                      Pos_list_1=self.Pose * 1000,
                                                                      particles_5=self.particles,
                                                                      weights_5=self.weights)

                            # 更新粒子，筛掉低权重粒子
                            w_thresh = 1.0 / (len(self.particles) * 2)
                            mask = self.weights >= w_thresh
                            self.particles_filter = self.particles[mask]
                            self.weights_filter = self.weights[mask]
                            std = np.std(self.particles_filter, axis=0)
                            if std[0] < 0.8 and std[1] < 0.8:
                                self.N = 4  ## 当粒子的范围足够小时，进入装配阶段
                                self.get_logger().info('The position is accurate enough!')
                                self.get_logger().info(f'The current particles distribution is {self.particles_filter}')
                            else:
                                self.particles, self.weights = self.M.resample_particles(self.particles_filter,
                                                                                         self.weights_filter,
                                                                                         particles_remain)

                            self.get_logger().info('The weights has update!!')
                            self.get_logger().info(f'len(particles)={len(self.particles)}')
                            self.get_logger().info(f'contact state:N={self.N}')
                            # self.get_logger().info(f'range:x_min={self.M.x_min},x_max={self.M.x_max}')
                            ##保存相关数据进行分析
                            np.savez(f'contact_slam_N1data_2511101625_{str(self.num)}.npz', particles=self.particles,
                                     weights=self.weights, Pose_list=self.Pose, particles_filter=self.particles_filter,
                                     Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                                     xs=self.xs, ys=self.ys)
                            self.get_logger().info(f'The N1 data has been saved!')
                            self.num += 1  ##这是一个计算机器人进入几次力调整的标志量

                        self.initial = None  ##此时再接收初始位置
                        self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
                        self.sum = np.zeros(3, dtype=np.float64)
                        self.count = 0
                        self.pos = 0  ##这些计算平均力的值也要归零

                    if self.i % 75 == 0:
                        self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                        self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                        self.get_logger().info(
                            f'delta_x= {delta_x}, delta_y= {delta_y}, delta_z= {delta_z}')
                        self.get_logger().info(
                            f'v_vec= {self.v_vec}, f_projec= {f_projec}, f_projec_mean= {f_projec_mean}')
            else:
                ##发布A_star得到的期望目标点，执行轨迹规划的结果
                if self.X is not None:
                    if self.n_now < self.X.shape[0]:  ##轨迹追踪还没有进行完
                        current_pose = np.array(
                            [self.current.position.x, self.current.position.y, self.current.position.z])
                        last_pose = np.array([self.X[self.n_now], self.Y[self.n_now], self.Z[self.n_now]])
                        ## 姿态误差的计算
                        current_quat = np.array([self.current.orientation.x, self.current.orientation.y,
                                                 self.current.orientation.z, self.current.orientation.w])
                        last_quat = self.Q[self.n_now]
                        if (np.linalg.norm(current_pose - last_pose) < 0.0015 and
                                abs(np.arccos(np.clip(np.dot(current_quat, last_quat), -1.0,
                                                      1.0)) * 2.0) < 0.002):  ##与上一个目标点的误差足够小，开始发布下一个目标点
                            self.n_now += 1
                            if self.n_now < self.X.shape[0]:
                                next_pose = np.array([self.X[self.n_now], self.Y[self.n_now], self.Z[self.n_now]])
                                self.v_vec = next_pose - last_pose  ##运动方向矢量
                        else:
                            self.v_vec = last_pose - current_pose
                        ##合力在运动方向上的分量
                        f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                        if self.N == 0 or self.N == 4:
                            f_ref = 2.0
                        elif self.N == 3:
                            f_ref = 3.0
                        else:
                            f_ref = f_z

                        if np.linalg.norm(f_projec) > self.mu * f_ref and (f_vec @ self.v_vec) < 0 and np.linalg.norm(
                                f_projec) > 0.2 \
                                and f_ref > 1.0:
                            self.force_control_flag = True  ##接触力超出阈值，进入接触力调整的阶段
                            ## 收集当前的接触状态信息
                            self.z = 1
                            self.Z_obs.append(self.z)
                            self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                            self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                            ## 重置以接收新目标（但先不接收初始位置）
                            self.get_logger().info('Terminate and change contact force!')
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        else:  ## 接触力没有超出阈值
                            ## 收集当前的接触状态信息
                            if self.N == 0:  ##处于竖直向下运动阶段
                                self.z = 0
                                if self.i % 10 == 0:
                                    self.Z_obs.append(self.z)
                                    self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                    self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                            elif self.N == 1:  ##处于水平运动阶段
                                if f_z > 0.4:  ## 竖直方向仍然存在接触
                                    self.z = 0
                                    if self.i % 10 == 0:
                                        self.Z_obs.append(self.z)
                                        self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                        self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                elif f_z < 0.4 and self.terminate_flag == False:
                                    self.terminate_flag = True
                                    self.get_logger().info('Get into contact test phase!')
                            elif self.N == 4:  ## 估计收敛，运动搜孔
                                ##由于dz_flag=1的执行过程已经被排除了，此时只需要考虑dz_flag=2
                                if f_z < 0.9 and self.dz_flag == 2:  ## 落入孔区域（边缘可能有接触）
                                    if abs(f_y) > 1.0 or abs(f_x) > 0.9:  ##横向有约束
                                        self.force_control_flag = True  ##接触力超出阈值，进入接触力调整的阶段
                                        ## 收集当前的接触状态信息
                                        self.z = 1
                                        self.Z_obs.append(self.z)
                                        self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                        self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                        ## 重置以接收新目标（但先不接收初始位置）
                                        self.get_logger().info('Inside the hole range and constrained!')
                                        self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                        self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                        self.total_time = 0.0
                                        self.n_now = 0
                                        # self.X, self.Y, self.Z = None, None, None
                                        self.force_inside_flag = False
                                    else:  ## 横向无约束
                                        if self.N_ite >= 2:
                                            self.force_control_flag = True  ##进入接触力调整的阶段
                                            self.z = 1
                                            self.Z_obs.append(self.z)
                                            self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                            self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                            ## 重置以接收新目标（但先不接收初始位置）
                                            self.get_logger().info('Inside the hole range but not constrained!')
                                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                            self.total_time = 0.0
                                            self.n_now = 0
                                            # self.X, self.Y, self.Z = None, None, None
                                            self.force_inside_flag = False
                                        else:
                                            self.z = 0
                                            if self.i % 10 == 0:
                                                self.Z_obs.append(self.z)
                                                self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                                self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                else:  ## 竖向仍然接触
                                    self.z = 0
                                    if self.i % 10 == 0:
                                        self.Z_obs.append(self.z)
                                        self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                        self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)

                            else:
                                self.z = 0
                                if self.i % 10 == 0:
                                    self.Z_obs.append(self.z)
                                    self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                    self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)

                            if self.n_now < self.X.shape[0]:  ##轨迹还未执行完，继续执行轨迹
                                if self.terminate_flag:  ##进入接触状态核查阶段（核查是否落入了孔区域，是否超出了运动区域）
                                    # if self.v_pred[1] == 1: ##核查是否落入了孔区域
                                    #     dy = 0.005
                                    # else:
                                    #     dy = -0.005
                                    self.cmd.position.x += delta_x / self.freq
                                    self.cmd.position.y += delta_y / self.freq
                                    self.cmd.position.z += (delta_z / self.freq - 0.001 / self.freq)
                                    self.terminate_num += (delta_z / self.freq - 0.001 / self.freq)  ## 计算累积向下运动的距离，单位为m
                                    print(f'terminate_num = {self.terminate_num}')
                                    if f_z > 1.0:  ##还存在接触
                                        self.terminate_flag = False
                                        self.terminate_num = 0.0
                                        self.get_logger().info('F_z>1 and get out the contact test phase!')
                                    else:  ##纵向接触力不足，说明落入了孔区域或者超出运动区域
                                        if abs(self.v_pred[1]) == 1 and abs(f_y) > 1.0:
                                            self.dz_flag = 1
                                        if abs(self.v_pred[0]) == 1 and abs(f_x) > 0.9:
                                            self.dz_flag = 1
                                        if self.dz_flag == 1:  # 横向受到了约束，说明落入了孔约束，下一步尝试向下插入（需进一步完善，比如x方向落入孔区域怎么办）
                                            self.N = 3  ## 此时还未收敛，尝试孔内约束是否正对孔中心
                                            self.dz_flag = 1  ##下一步准备向下运动插入
                                            self.initial = None
                                            self.total_time = 0.0
                                            self.n_now = 0
                                            self.X, self.Y, self.Z = None, None, None
                                            self.force_inside_flag = False
                                            self.z = 1
                                            self.Z_obs.append(self.z)
                                            self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                            self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                            ## 将数据备份
                                            self.Z_obs_pre = self.Z_obs
                                            self.Pose_pre = self.Pose
                                            self.v_pred_pre = self.v_pred
                                            self.terminate_flag = False
                                            self.terminate_num = 0
                                            self.get_logger().info('inside the hole range!')
                                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                        # elif f_z > 1.0:
                                        #     self.terminate_flag = False
                                        #     self.terminate_num = 0
                                        #     self.get_logger().info('F_z>1 and get out the contact test phase!')
                                        elif self.terminate_num < -0.025 and f_z < 0.2:  # 横向没有受到约束，纵向也没有受到约束，判定为超出范围
                                            self.z = -1
                                            self.Z_obs.append(self.z)
                                            self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                            self.F_x_N.append(f_x), self.F_y_N.append(f_y), self.F_z_N.append(f_z)
                                            self.terminate_flag = False
                                            self.terminate_num = 0
                                else:
                                    self.cmd.position.x = self.X[self.n_now] + delta_x / self.freq
                                    self.cmd.position.y = self.Y[self.n_now] + delta_y / self.freq
                                    if self.N == 1:
                                        self.cmd.position.z += delta_z / self.freq
                                    else:
                                        self.cmd.position.z = self.Z[self.n_now] + delta_z / self.freq
                                    # orientation 可插值或保持初始
                                    self.cmd.orientation.x = self.Q[self.n_now][0]
                                    self.cmd.orientation.y = self.Q[self.n_now][1]
                                    self.cmd.orientation.z = self.Q[self.n_now][2]
                                    self.cmd.orientation.w = self.Q[self.n_now][3]
                                if self.i % 100 == 0:
                                    self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                    self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                    self.get_logger().info(
                                        f'delta_x= {delta_x}, delta_y= {delta_y}, delta_z= {delta_z}, '
                                        f'v_vec= {self.v_vec}, f_projec= {f_projec}')
                            else:
                                self.cmd.position.x += delta_x / self.freq
                                self.cmd.position.y += delta_y / self.freq
                                self.cmd.position.z += delta_z / self.freq

                            ##如果状态为-1,则返回出发点重新探索，同时更新权重和粒子
                            if self.z == -1:
                                self.force_control_flag = True  ##也进入接触力调整和推理阶段
                                ## 重置以接收新目标（但先不接收初始位置）
                                self.get_logger().info('Beyond the contact range')
                                self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                                self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                                self.total_time = 0.0
                                self.n_now = 0
                                self.X, self.Y, self.Z = None, None, None
                                self.force_inside_flag = False
                                self.N = 2  # 这个标志代表重新返回上一次出发点

                    else:  ##轨迹完全追踪完了，说明此时的阶段为N=2或N=3
                        if self.N == 4 and self.dz_flag == 0:  ## N4模式下，此时移动到了目标位置的上方，但还没有插入
                            self.get_logger().info('insert but haven\'t contact with surface')
                            ## 重置以接收新目标
                            self.dz_flag = 1  ##下一步准备向下运动运动
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        elif self.N == 4 and self.dz_flag == 2:  ## N4模式下，此时说明经过横向移动之后还没有搜到孔
                            self.get_logger().info('Haven\'t find the hole range')
                            ## 重置以接收新目标
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        elif self.N == 2:  ##说明此时是N=2
                            ## 重置以接收新目标
                            self.get_logger().info('N2 Trajectory complete and return to the initial pose.')
                            self.N = 0  ##重新向下运动进行探索
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        else:
                            self.N = 5
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                else:
                    if self.N < 5:
                        self.As_planner()
                    else:
                        self.get_logger().info('The Planning Process has ended!')

            self.pose_pub.publish(self.cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MinimumJerkPosePlanner()

    # 设置拉杆：假定先设置 total_time，可通过 launch 参数或 Node 参数改动
    # node.total_time = 8.0  # 例如 5 秒完成轨迹
    node.target = Pose()
    node.target.position.x = 0.0
    node.target.position.y = 0.0
    node.target.position.z = 0.0

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        np.savez(r'Force_data_2511101625.npz', Force_left=node.Force_left, Force_right=node.Force_right,
                 Matrix_left=node.Matrix_left, Matrix_right=node.Matrix_right, Force_dis_left=node.Force_dis_left,
                 Force_dis_right=node.Force_dis_right,
                 F_x=node.F_x, F_y=node.F_y, F_z=node.F_z, R_x=node.R_x, R_y=node.R_y, R_z=node.R_z,
                 Pose=node.Pose_all, Time=node.Time)
        print('save Data!')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
