# import math
import math
import pickle

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from scipy.interpolate import splrep, splev
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.transform import Rotation as R, Slerp

# from .motion_planning import Motion_planning
from .motion_planning_3 import Motion_planning
from tutorial_interfaces.msg import Cloud, Array3
from .tac3d_test import gripper_force_direct, contact_point_estimate, select_top_n, estimate_rigid_transform_kabsch

import symforce
symforce.set_epsilon_to_invalid()

from symforce import geo
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.optimizer import OptimizerParams
"""
blind pushing的正式版本，但是推块过程中的过程还无法实现自主规划。
"""


class MinimumJerkPosePlanner(Node):
    def __init__(self):
        super().__init__('min_jerk_pose_planner_py')
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

        self.P_l = None  # 左手Tac3D marker点位置分布数据
        self.P_r = None  # 右手Tac3D marker点位置分布数据
        self.D_l = None  # 左手marker点位移分布数据
        self.D_r = None  # 右手marker点位移分布数据
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
        self.Contact = []
        self.target_sure = False

        self.force_inside_flag = False
        self.force_control_flag = False
        self.move_hori_flag = False
        self.i = 0
        self.N = 0
        self.n_now = 0
        self.terminate_flag = False
        self.terminate_num = 0

        self.window_size = 6
        self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.count = 0
        self.pos = 0

        self.m = 0
        self.F_x_1 = np.zeros(0)
        self.F_y_1 = np.zeros(0)
        self.F_z_1 = np.zeros(0)
        self.Force_left_1 = np.zeros(0)
        self.Force_right_1 = np.zeros(0)
        self.Matrix_left_1 = np.zeros(0)
        self.Matrix_right_1 = np.zeros(0)
        self.f_x_mean = None
        self.f_y_mean = None
        self.f_z_mean = None
        self.force_left_mean = None
        self.force_right_mean = None
        self.matrix_left_mean = None
        self.matrix_right_mean = None
        method_results = np.load("./method_results.npz")
        # 得到的第一组delta值进行后续C值计算（也可选择其他组或所有组的平均值）
        self.delta_L = method_results["delta_L_list"][0]
        self.delta_R = method_results["delta_R_list"][0]

        ##factor graph
        self.diff_points = None
        self.row_index = None
        self.refer_points = None
        self.world_to_gripper_translation = None
        self.gripper_to_sensor_translation = None
        self.sensor_to_tool_translation = None
        self.sensor_sense_translation = None
        self.sensor_to_tool_permant = None
        self.gripper_pose_prior = None
        self.R_left = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])  ## 将点的坐标从传感器坐标系转换到夹爪坐标系
        self.R_right = np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
        ##构造因子
        self.factors = []
        self.values = Values()
        self.curr_optimized_values = Values()
        self.curr_optimized_factors = []
        self.params = OptimizerParams(verbose=False)
        self.k = 1
        self.j = 1 ##N3阶段不同的轨迹计数

        ##进行主动探索所需要的参数
        self.peaks = 0  ##峰值数目
        self.max_points = []  ##权重符合条件的粒子位置和权重
        self.obstacles_all = [
            np.array([[0.0, 0.0], [0.0, -180.0], [20.0, -180.0], [20.0, -20.0], [160.0, -20.0],
                      [160.0, 0.0]])
        ]
        self.Movable_objects = [
            np.array([[-4.5, 85.0], [-4.5, -85.0], [4.5, -85.0], [4.5, 85.0]], dtype=float),  # 小矩形1
        ]
        self.block = [
            np.array([[-21.65,12.5], [-21.65, -12.5], [0.0, -25.0], [21.65, -12.5], [21.65, 12.5], [0.0, 25.0]])
        ] ## 待推动block的尺寸
        self.block_ref_point = np.array([0.0, 0.0]) ## block的中心参考点，用于确定位置
        self.particles = None
        self.z = None
        self.weights = None
        self.Z_obs = None
        self.num = 0  ##这是一个计算机器人进入几次力调整的次数的量
        self.Pose = np.zeros(0)
        self.Pose_contact = np.zeros(0)
        self.block_range = np.zeros(0)
        self.v_pred = np.zeros(0)
        # ## 二指插座的间距设置
        # curve_dis = np.array([10.4, 35.7, 15.5, 15.5])  ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
        ## 三指插座的间距设置
        curve_dis = np.array([125.5, 125.5, 85.0, 85.0])  ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
        self.M = Motion_planning(d=2.0, padding=curve_dis)  # 这里的单位是mm
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
            Cloud, '/forces_l', self.on_F_l, 1)
        self.F_r_sub = self.create_subscription(
            Cloud, '/forces_r', self.on_F_r, 1)
        self.Fr_l_sub = self.create_subscription(
            Array3, '/resultant_force_l', self.on_Fr_l, 1)
        self.Fr_r_sub = self.create_subscription(
            Array3, '/resultant_force_r', self.on_Fr_r, 1)
        self.Mr_l_sub = self.create_subscription(
            Array3, '/resultant_moment_l', self.on_Mr_l, 1)
        self.Mr_r_sub = self.create_subscription(
            Array3, '/resultant_moment_r', self.on_Mr_r, 1)

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
        # self.get_logger().info(f'D_l array shape: {xyz.shape}')

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

    def on_Fr_l(self, msg: Cloud):
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

    def on_Fr_r(self, msg: Cloud):
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

    def on_Mr_l(self, msg: Cloud):
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

    def on_Mr_r(self, msg: Cloud):
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
                    if -10 < self.Fr_r[0, 2] < -9 and -10 < self.Fr_l[0, 2] < -9 and self.force_inside_flag == False:
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
                    ##因子图的准备工作
                    self.current = msg
                    if self.refer_points is None:
                        self.row_index, _ = select_top_n(self.D_l, N=10, by_abs=True)
                        self.refer_points = self.P_l[self.row_index, :]
                        self.refer_points = np.dot(self.R_left, self.refer_points.T).T
                    else:
                        # 每一时刻夹爪的位置先验，由机械臂解算传回
                        self.gripper_pose_prior = geo.V3(msg.position.x*1000, msg.position.y*1000,
                                                         msg.position.z*1000) ##转化成mm
                        if all(v is not None for v in (self.P_l, self.P_r, self.D_l, self.D_r)):
                            self.sensor_to_tool_permant = geo.M44(1, 0, 0, 0,
                                                                  0, 1, 0, 0,
                                                                  0, 0, 1, -160.0,
                                                                  0, 0, 0, 1)  # 工具坐标系相对于传感器坐标系的转换先验（定值，需事先标定）
                            self.diff_points = self.P_l[self.row_index, :]
                            self.diff_points = np.dot(self.R_left, self.diff_points.T).T
                            R, T = estimate_rigid_transform_kabsch(self.refer_points, self.diff_points)
                            self.sensor_sense_translation = geo.M44(R[0, 0], R[0, 1], R[0, 2], T[0],
                                                                    R[1, 0], R[1, 1], R[1, 2], T[1],
                                                                    R[2, 0], R[2, 1], R[2, 2], T[2],
                                                                    0, 0, 0, 1)  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
                            self.sensor_to_tool_translation = self.sensor_sense_translation * self.sensor_to_tool_permant
                            self.gripper_to_sensor_translation = geo.M44(1, 0, 0, 0,
                                                                         0, 1, 0, 0,
                                                                         0, 0, 1, -200.0,
                                                                         0, 0, 0, 1)  # 传感器坐标系相对于夹爪（机械臂末端）坐标系的转换（定值）
                            self.world_to_gripper_translation = geo.M44(1, 0, 0, msg.position.x*1000,
                                                                        0, 1, 0, msg.position.y*1000,
                                                                        0, 0, 1, msg.position.z*1000,
                                                                        0, 0, 0, 1)

                    if self.m < 20:
                        f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
                        self.F_x_1 = np.append(self.F_x_1, f_x)
                        self.F_y_1 = np.append(self.F_y_1, f_y)
                        self.F_z_1 = np.append(self.F_z_1, f_z)
                        self.Force_left_1 = np.append(self.Force_left_1, self.Fr_l)
                        self.Force_right_1 = np.append(self.Force_right_1, self.Fr_r)
                        self.Matrix_left_1 = np.append(self.Matrix_left_1, self.Mr_l)
                        self.Matrix_right_1 = np.append(self.Matrix_right_1, self.Mr_r)
                        self.m += 1
                    elif self.m == 20:
                        self.f_x_mean = np.mean(self.F_x_1)
                        self.f_y_mean = np.mean(self.F_y_1)
                        self.f_z_mean = np.mean(self.F_z_1)
                        self.force_left_mean = np.mean(self.Force_left_1, axis=0)
                        self.force_right_mean = np.mean(self.Force_right_1, axis=0)
                        self.matrix_left_mean = np.mean(self.Matrix_left_1, axis=0)
                        self.matrix_right_mean = np.mean(self.Matrix_right_1, axis=0)
                        self.get_logger().info(
                            f'The mean initial force is f_x={self.f_x_mean}, f_y={self.f_y_mean}, f_z={self.f_z_mean}')
                        self.m += 1
                    else:
                        # if self.i % 50 == 0:
                        #     self.contact_optimizer()
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
            if self.N == 0:  ## 从出发区域开始向目标区域推块
                if self.target_sure == False: ##false说明尚在刚开始的推块阶段
                    Points_recover.append(start + np.array([-0.1, 0.0, 0.0]))  # 这里的单位是m
                    Points_recover.append(start + np.array([-0.2, 0.0, 0.0]))
                    self.v_pred = np.array([-1, 0])  ##末端只在竖直方向上运动，横向速度均为0
                # else: ##True说明此时不是初始推块阶段，至少已经经历过一次姿态转换
                #     # ===== 新增: 等待键盘输入 'q' 才继续 =====
                #     self.get_logger().info("Change v_pred and Press 'q' and Enter to continue...")
                #     while True:
                #         user_in = input()
                #         if user_in == 'w':
                #             try:
                #                 new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                #                 new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                #                 distance = float(input("请输入运动的距离: "))
                #                 self.v_pred = np.array([new_val_0, new_val_1])
                #                 self.get_logger().info(f"self.v_pred 已更新为 {self.v_pred}")
                #             except ValueError:
                #                 self.get_logger().warn("输入无效，请输入数值。")
                #         if user_in == 'q':
                #             break
                #         else:
                #             print("无效输入，请输入 'q' 或 'w'")
                #     delta = self.v_pred * distance ##这里的单位为m
                #     # 加入到轨迹规划中
                #     Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.0]))
                #     Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = []  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.m = 0

            elif self.N == 1:  ##碰到障碍物，变为N1,此时开始进行探索
                self.dz_flag = 0
                # 检测峰值，选择相应的方向
                # peaks = self.M.find_local_maximum(self.particles, self.weights, radius=5)
                # self.get_logger().info(f'len(peaks) = {len(peaks)}')
                # self.v_pred = self.M.gain_information(peaks, self.Movable_objects)
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
                    elif user_in == 'c':
                        try:
                            new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                            new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                            self.dhori = float(input("请输入横向运动的数值: "))
                            self.v_pred = np.array([new_val_0, new_val_1])
                            self.get_logger().info(f"选定的运动方向为：{self.v_pred}")
                            self.dz_flag = 1
                        except ValueError:
                            self.get_logger().warn("输入无效，请输入数值。")

                    else:
                        print("无效输入，请输入 'q' 或 'w'")
                # 将运动方向转化为运动距离
                if self.dz_flag == 1:
                    delta = self.v_pred * self.dhori
                    # 加入到轨迹规划中
                    Points_recover.append(start + np.array([0.0, 0.0, 0.05]))
                    Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.05]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.05]))
                    self.get_logger().warn("完成\"跳岛\"轨迹规划")
                else:
                    delta = self.v_pred * 0.3  ##这里的单位为m
                    # 加入到轨迹规划中
                    Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = []  ##机械臂末端运动的位置（后续用于更新权重和定位）

            elif self.N == 2: ## 超出探索范围，开始返回上一次初始位置
                current_pose = np.array(
                    [self.current.position.x, self.current.position.y])
                delta = -(current_pose - self.Pose[0])  ##这里单位为m
                self.particles = self.particles + delta * 1000  ##更新粒子的位置，将粒子的分布也返回上一次出发点（单位转化成mm）
                self.get_logger().info('Move the particles to last initial position!')
                Points_recover.append(start + np.array([0.5 * delta[0], 0.5 * delta[1], 0.07]))  ##适当向上运动
                Points_recover.append(start + np.array([delta[0], delta[1], 0.07]))
                self.v_pred = np.array([0, 0])  ##这里是回归上一时刻的出发位置。
                self.z = 0  ## 触觉信号置0,重新开始观测
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = []  ##机械臂末端运动的位置（后续用于更新权重和定位）
                self.N = 0  ##重新向下运动进行探索

            elif self.N == 3: ## 进入姿态调整阶段，包括上升、平移和旋转（目前版本：后续的所有动作都包含）
                delta = np.array([0.0, 0.0])
                if self.j == 1:
                    delta = np.array([85.0-21.5*2+5, 50]) * 0.001
                    qd = q0  ## 调姿
                    Points_recover.append(start + np.array([0.0, 0.0, 0.06]))  # 这里的单位是m，首先上升
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.06]))  ##然后平移
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 2:
                    delta = np.array([0.1, 0.1]) * 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 3:
                    delta = np.array([0.0, 0.1]) * 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, -0.06]))
                    Points_recover.append(start + np.array([delta[0], delta[1], -0.06]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 4:
                    delta = np.array([0.0, -170.0]) * 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 5:
                    delta = np.array([0.0, -75.0]) * 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([0.0, 0.0, 0.06]))  # 这里的单位是m，首先上升
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.06]))  ##然后平移
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 6:
                    delta = np.array([0.0, -0.1]) * 0.001
                    qd = np.array([0.0,
                                   1.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 7:
                    delta = np.array([0.0, -0.1]) * 0.001
                    qd = np.array([0.0,
                                   1.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, -0.06]))
                    Points_recover.append(start + np.array([delta[0], delta[1], -0.06]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 8:
                    delta = np.array([-21.25*2-85.0-20.0, -0.0]) * 0.001
                    qd = np.array([0.0,
                                   1.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 9:
                    delta = np.array([-80.0, 0.0]) * 0.001
                    qd = np.array([0.0,
                                   1.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([0.0, 0.0, 0.06]))  # 这里的单位是m，首先上升
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.06]))  ##然后平移
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 10:
                    delta = np.array([0.1, 0.1])* 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 11:
                    delta = np.array([0.1, 0.1])* 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, -0.06]))
                    Points_recover.append(start + np.array([delta[0], delta[1], -0.06]))
                    self.get_logger().info(f'N3 trajectory, j={self.j}')
                elif self.j == 12:
                    delta = np.array([0.0, 120.0+85.0]) * 0.001
                    qd = np.array([math.sqrt(2) / 2.0,
                                   math.sqrt(2) / 2.0,
                                   0.0,
                                   0.0])  ## 调姿
                    Points_recover.append(start + np.array([delta[0]*0.5, delta[1]*0.5, 0.0]))
                    Points_recover.append(start + np.array([delta[0], delta[1], 0.0]))

                self.v_pred = np.array([0, 0])  ##末端只在竖直方向上运动，横向速度均为0
                self.Z_obs = []  ##触觉信号收集list
                self.Pose = []  ##机械臂末端运动的位置（后续用于更新权重和定位）

            elif self.N > 3:
                self.get_logger().info('End the trajectory!')
            ## 姿态的规划
            if self.N != 3: ##模式不为3时，不需要调姿
                qd = np.array([0.0,
                               1.0,
                               0.0,
                               0.0])

            for j in range(len(Points_recover) - 1):
                self.total_time += max(np.linalg.norm(Points_recover[j + 1] - Points_recover[j]) / 0.01,
                                       abs(np.arccos(np.clip(np.dot(q0, qd), -1.0, 1.0)) * 2.0) / 0.055)

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
            Fr_r_1 = self.Fr_r - self.force_right_mean
            Fr_l_1 = self.Fr_l - self.force_left_mean
            Mr_r_1 = self.Mr_r - self.matrix_right_mean
            Mr_l_1 = self.Mr_l - self.matrix_left_mean
            self.F_x.append(f_x)
            self.F_y.append(f_y)
            self.F_z.append(f_z)
            self.R_x.append(r_x), self.R_y.append(r_y), self.R_z.append(r_z)
            self.Pose_all.append(np.array(
                [self.current.position.x, self.current.position.y, self.current.position.z]))
            self.Time.append(current_time)

            ##计算接触位置
            F_C = np.array([f_x, f_y, f_z]).reshape(1, 3)
            force_left_w = np.array([-Fr_r_1[0, 1], Fr_r_1[0, 2], -Fr_r_1[0, 0]]).reshape(1,3)
            force_right_w = np.array([Fr_l_1[0, 1], -Fr_l_1[0, 2], -Fr_l_1[0, 0]]).reshape(1, 3)
            matrix_left_w = np.array(
                [-Mr_r_1[0, 1], Mr_r_1[0, 2], -Mr_r_1[0, 0]]).reshape(1, 3)
            matrix_right_w = np.array([Mr_l_1[0, 1], -Mr_l_1[0, 2], -Mr_l_1[0, 0]]).reshape(
                1, 3)
            Contact_c = contact_point_estimate(FL=force_left_w, FR=force_right_w, ML=matrix_left_w,
                                               MR=matrix_right_w, FC=F_C, delta_L=self.delta_L, delta_R=self.delta_R)
            self.Contact.append(Contact_c)

            ##计算合力矢量
            f_vec = np.array([f_x, f_y, f_z])

            ##计算偏移量(刚度越大，偏移量应该越小)
            delta_z = np.sign(f_z - 2.1) * min(0.0032, 0.0032 * abs(f_z - 2.1)) if abs(f_z) > 0.05 else 0.0
            delta_x = np.sign(f_x - 0.05) * min(0.005, 0.05 * abs(f_x - 0.05)) if abs(f_x) > 0.05 else 0.0
            delta_y = np.sign(f_y - 0.1) * min(0.005, 0.005 * abs(f_y - 0.1)) if abs(f_y) > 0.1 else 0.0

            if self.force_control_flag:  ##接触力调整与推理阶段，保持一个恒定的接触力
                self.cmd.position.x += delta_x / self.freq
                self.cmd.position.y += delta_y / self.freq
                self.cmd.position.z += delta_z / self.freq


                if self.N == 3:
                    # delta = self.v_pred * self.dhori
                    # self.particles = self.particles + delta
                    # particles_last = np.zeros(0)
                    # if self.Z_obs[-1] == 0:
                    #     for p in self.particles:
                    #         if p[0] < self.xs[0] or p[0] > self.xs[-1] or p[1] < self.ys[0] or p[1] > self.ys[-1]:
                    #             particles_last = np.append(particles_last, p).reshape(-1, 2)
                    # elif self.Z_obs[-1] == 1:
                    #     for p in self.particles:
                    #         if self.xs[0] < p[0] < self.xs[-1] and self.ys[0] < p[1] < self.ys[-1]:
                    #             particles_last = np.append(particles_last, p).reshape(-1, 2)
                    # self.weights = np.ones(len(particles_last)) / len(particles_last)
                    # self.particles = particles_last
                    # ##保存相关数据进行分析
                    # np.savez(f'contact_slam_N3data_2509071158_{str(self.num)}.npz', particles=self.particles,
                    #          weights=self.weights, Pose_list=self.Pose,
                    #          Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                    #          xs=self.xs, ys=self.ys)
                    self.get_logger().info(f'The N3 data has been saved!')
                    self.j += 1

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
                    # # self.get_logger().info(f'range:x_min={self.M.x_min},x_max={self.M.x_max}')
                    # # 更新权重
                    # self.weights = self.M.weights_updates_dis(action=self.v_pred,
                    #                                           Movable_objects_1=self.Movable_objects,
                    #                                           z_obs_list=self.Z_obs,
                    #                                           Pos_list_1=self.Pose * 1000, particles_5=self.particles,
                    #                                           weights_5=self.weights)
                    # self.get_logger().info('The weights has update!!')
                    # self.get_logger().info(f'len(particles)={len(self.particles)}')
                    # self.get_logger().info(f'contact state:N={self.N}')
                    ##保存相关数据进行分析
                    # np.savez(f'contact_slam_N2data_2509071158_{str(self.num)}.npz', particles=self.particles,
                    #          weights=self.weights, Pose_list=self.Pose,
                    #          Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                    #          xs=self.xs, ys=self.ys)
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
                    if self.N == 0:  ## 推块的参考力应和遇到障碍物相区别
                        f_ref = 0.2
                    else:
                        f_ref = 0.2  ##根据实际情况调整

                    if self.mu * f_ref * 0.3 < np.linalg.norm(
                            f_projec_mean) < self.mu * f_ref * 0.5:  ##分量的大小最近若干个值均在范围内，则认为接触力调整完成。
                        self.force_control_flag = False
                        self.get_logger().info('Contact force within threshold!')
                        ##进入粒子权重更新阶段
                        if self.N == 0: ##推块过程碰到障碍物停下
                            # if self.num == 0:
                            #     # 初始化粒子分布
                            #     self.get_logger().info(f'v_pred={self.v_pred}, z={self.Z_obs[-1]}')
                            #     particle_range = self.M.main_line_perception(self.Z_obs,
                            #                                                  self.v_pred, self.Movable_objects)
                            #     self.particles, self.weights = self.M.generate_particles_in_polygon(particle_range,
                            #                                                                         N=2000)
                            #     ##此时的粒子分布并不是最终的粒子分布，中间还隔了一个block。
                            #     block_polys = [Polygon(p) for p in self.block]
                            #     region = unary_union(block_polys)  # 取并集，得到完整区域
                            #     # 得到边界框
                            #     rx_min, ry_min, rx_max, ry_max = region.bounds
                            #     block_dx = rx_max - rx_min
                            #     block_dy = ry_max - ry_min
                            #     ##平移粒子
                            #     self.particles = self.particles + (-self.v_pred) * np.array([block_dx, block_dy])
                            #     # 更新权重
                            #     self.weights = self.M.weights_updates_dis(action=self.v_pred,
                            #                                               Movable_objects_1=self.Movable_objects,
                            #                                               z_obs_list=self.Z_obs,
                            #                                               Pos_list_1=self.Pose * 1000,
                            #                                               particles_5=self.particles,
                            #                                               weights_5=self.weights)
                            #
                            self.N = 1 ##进入环境探索阶段
                            # self.get_logger().info('Transfer to exploration phase!')
                            # self.get_logger().info(f'len(particles)={len(self.particles)}')
                            # self.get_logger().info(f'contact state:N={self.N}')
                            # # self.get_logger().info(f'range:x_min={self.M.x_min},x_max={self.M.x_max}')
                            # ##保存相关数据进行分析
                            # np.savez(f'contact_slam_N0data_2509071158_{str(self.num)}.npz', particles=self.particles,
                            #          weights=self.weights, Pose_list=self.Pose,
                            #          Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                            #          xs=self.xs, ys=self.ys, block_range=self.block_range, Pose_contact=self.Pose_contact)
                            self.get_logger().info(f'The N0 data has been saved!')
                            self.num += 1  ##这是一个计算机器人进入几次力调整的标志量

                        elif self.N == 1:
                            # particle_range = self.M.main_line_perception(self.Z_obs,
                            #                                              self.v_pred, self.Movable_objects)
                            #
                            # self.particles, self.weights = self.M.update_particles(self.particles, self.weights,
                            #                                                         particle_range,
                            #                                                         shift_3=(self.Pose[-1] -
                            #                                                                 self.Pose[0]) * 1000)
                            #
                            # # 更新权重
                            # self.weights = self.M.weights_updates_dis(action=self.v_pred,
                            #                                           Movable_objects_1=self.Movable_objects,
                            #                                           z_obs_list=self.Z_obs,
                            #                                           Pos_list_1=self.Pose * 1000,
                            #                                           particles_5=self.particles,
                            #                                           weights_5=self.weights)
                            # if len(self.particles) < 40:
                            #     self.N = 3 ## 粒子集中在特定区域后，进入调姿阶段
                            self.N = 3
                            # self.get_logger().info('The weights has update!!')
                            # self.get_logger().info(f'len(particles)={len(self.particles)}')
                            # self.get_logger().info(f'contact state:N={self.N}')
                            # # self.get_logger().info(f'range:x_min={self.M.x_min},x_max={self.M.x_max}')
                            # ##保存相关数据进行分析
                            # np.savez(f'contact_slam_N1data_2509071158_{str(self.num)}.npz', particles=self.particles,
                            #          weights=self.weights, Pose_list=self.Pose,
                            #          Z_list=self.Z_obs, Time=self.Time, v_pred=self.v_pred, scene=self.scene,
                            #          xs=self.xs, ys=self.ys, block_range=self.block_range,
                            #          Pose_contact=self.Pose_contact)
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
                        if (np.linalg.norm(current_pose - last_pose) < 0.0025 and
                                abs(np.arccos(np.clip(np.dot(current_quat, last_quat), -1.0,
                                                      1.0)) * 2.0) < 0.0035):  ##与上一个目标点的误差足够小，开始发布下一个目标点
                            self.n_now += 1
                            if self.n_now < self.X.shape[0]:
                                next_pose = np.array([self.X[self.n_now], self.Y[self.n_now], self.Z[self.n_now]])
                                self.v_vec = next_pose - last_pose  ##运动方向矢量
                        else:
                            self.v_vec = last_pose - current_pose
                        ##合力在运动方向上的分量
                        f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                        if self.N == 0: ## 推块的参考力应和遇到障碍物相区别
                            f_ref = 0.2
                        else:
                            f_ref = 0.3 ##根据实际情况调整

                        if np.linalg.norm(f_projec) > self.mu * f_ref and (f_vec @ self.v_vec) < 0 and self.N != 3:
                            self.force_control_flag = True  ##接触力超出阈值，进入接触力调整的阶段
                            ## 收集当前的接触状态信息
                            self.z = 1  ##这里有问题
                            self.Z_obs.append(self.z)
                            self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                            ## 重置以接收新目标（但先不接收初始位置）
                            self.get_logger().info('Terminate and change contact force!')
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                            self.get_logger().info(f'Contact= {Contact_c}')
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        else:  ## 接触力没有超出阈值
                            ## 收集当前的接触状态信息
                            if self.N == 0:  ##处于水平推块阶段
                                self.z = 0
                                if self.i % 10 == 0:
                                    self.Z_obs.append(self.z)
                                    self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                                ## 判断是否接触block
                                if np.linalg.norm(f_projec) > 0.06 and (f_vec @ self.v_vec) < 0:
                                    self.block_range = np.array([[-4.5, -5.0], [-4.5, 85.0]]) ##block接触区域相对于objects的坐标. [(xmin,ymin),(xmax,ymax]
                                    self.Pose_contact = np.append(self.Pose_contact, current_pose[:2]).reshape(-1, 2)
                                    self.target_sure = True
                            elif self.N == 1:  ##处于水平运动阶段
                                self.z = 0
                                if self.i % 10 == 0:
                                    self.Z_obs.append(self.z)
                                    self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)
                            else:
                                self.z = 0
                                if self.i % 10 == 0:
                                    self.Z_obs.append(self.z)
                                    self.Pose = np.append(self.Pose, current_pose[:2]).reshape(-1, 2)

                            if self.n_now < self.X.shape[0]:  ##轨迹还未执行完，继续执行轨迹
                                if self.terminate_flag:  ##进入接触状态核查阶段
                                    self.cmd.position.x += delta_x / self.freq
                                    self.cmd.position.y += delta_y / self.freq
                                    self.cmd.position.z += delta_z / self.freq
                                else:
                                    self.cmd.position.x = self.X[self.n_now] + delta_x / self.freq
                                    self.cmd.position.y = self.Y[self.n_now] + delta_y / self.freq
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
                                    self.get_logger().info(f'Contact= {Contact_c}')
                            else:
                                self.cmd.position.x += delta_x / self.freq
                                self.cmd.position.y += delta_y / self.freq
                                self.cmd.position.z += delta_z / self.freq

                            # ##如果状态为-1,则返回出发点重新探索，同时更新权重和粒子
                            # if self.z == -1:
                            #     self.force_control_flag = True  ##也进入接触力调整和推理阶段
                            #     ## 重置以接收新目标（但先不接收初始位置）
                            #     self.get_logger().info('Beyond the contact range')
                            #     self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                            #     self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                            #     self.get_logger().info(f'Contact= {Contact_c}')
                            #     self.total_time = 0.0
                            #     self.n_now = 0
                            #     self.X, self.Y, self.Z = None, None, None
                            #     self.force_inside_flag = False
                            #     self.N = 2  # 这个标志代表重新返回上一次出发点

                    else:  ##轨迹完全追踪完了，说明探索失败了，需要重新探索
                        if self.N == 3:  ## N3模式下，没有接触（轨迹完整）说明可以进入推理阶段
                            self.force_control_flag = True  ##也进入接触力调整和推理阶段
                            ## 重置以接收新目标（但先不接收初始位置）
                            self.get_logger().info('N3 Trajectory complete.')
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                            self.get_logger().info(f'Contact= {Contact_c}')
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                        else:
                            if self.N == 0:
                                self.N = 3
                            ## 重置以接收新目标
                            self.get_logger().info('Trajectory complete.')
                            self.initial = None
                            self.total_time = 0.0
                            self.n_now = 0
                            self.X, self.Y, self.Z = None, None, None
                            self.force_inside_flag = False
                else:
                    if self.N < 4:
                        self.As_planner()
                    else:
                        self.get_logger().info('The Planning Process has ended!')

            self.pose_pub.publish(self.cmd)

    def contact_optimizer(self):
        ##夹爪先验位置约束
        if self.k == 1:
            self.values[f"g{self.k}"] = geo.V3(0.0, 0.0, 0.0)
        else:
            self.values[f"g{self.k}"] = self.values[f"g{self.k-1}"]
        def gripper_pose_error(gi: geo.V3) -> geo.V3:
            return gi - self.gripper_pose_prior
        self.factors.append(Factor(keys=[f"g{self.k}"], residual=gripper_pose_error))

        ##工具位置约束
        if self.k == 1:
            self.values[f"l{self.k}"] = geo.V3(0.0, 0.0, 0.0)
        else:
            self.values[f"l{self.k}"] = self.values[f"l{self.k - 1}"]
        def tool_pose_error(li: geo.V3) -> geo.V3:
            return (self.world_to_gripper_translation * self.gripper_to_sensor_translation * self.sensor_to_tool_translation * geo.V4(0, 0, 0, 1))[0:3] - li
        self.factors.append(Factor(keys=[f"l{self.k}"], residual=tool_pose_error))

        # ---- 优化器参数 ----
        params = OptimizerParams(verbose=False)

        # ---- 构建优化器 ----
        optimizer = Optimizer(
            factors=self.factors,
            optimized_keys=list(self.values.keys()),
            params=params,
            debug_stats=True,
        )
        result = optimizer.optimize(self.values)
        self.values = result.optimized_values

        # 打印当前状态
        g_curr = self.values[f"g{self.k}"]
        l_curr = self.values[f"l{self.k}"]
        print(f"t={self.k}, g={g_curr}, l={l_curr}")

        self.k += 1


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
        np.savez(r'Force_data_2509121022.npz', Force_left=node.Force_left, Force_right=node.Force_right,
                 Matrix_left=node.Matrix_left, Matrix_right=node.Matrix_right, Force_dis_left=node.Force_dis_left,
                 Force_dis_right=node.Force_dis_right,
                 F_x=node.F_x, F_y=node.F_y, F_z=node.F_z, R_x=node.R_x, R_y=node.R_y, R_z=node.R_z,
                 Pose=node.Pose_all, Time=node.Time, Contact=node.Contact)
        # 保存 Values 对象到文件
        with open("values_2509131804.pkl", "wb") as f:
            pickle.dump(node.values, f)
        print('save Data!')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
