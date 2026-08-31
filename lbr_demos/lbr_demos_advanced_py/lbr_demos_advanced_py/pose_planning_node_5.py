import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Vector3
import numpy as np
from .motion_planning import Motion_planning
import ros2_numpy as rnp
from tutorial_interfaces.msg import Cloud, Array3
from .tac3d_test import gripper_force_flow, gripper_force_direct, contact_point_estimate

"""
触觉顺应运动
运动规划的要求：先根据有限的关键点运动，当遇到某一方向力过大时，顺应减少运动直至停止，直到推理出新的运动规划关键点，然后继续运动
250818实验结果：接触后力过大，导致下一次规划后初始力也过大，需要在超过阈值到运行下一次规划之间有一个自适应调整的过程（收集机械臂本体力、触觉力）。
250819实验结果：接触力控制在了合理的范围内，下一步应改成任意方向的力的判断，然后开始进行轨迹规划。

本代码为可运行的触觉顺应轨迹规划v1.0，且能够确保运行过程中不会出现过大的速度变化，没有针对装配操作进行优化，主要用于微调机械臂初始位姿。
"""

class MinimumJerkPosePlanner(Node):
    def __init__(self):
        super().__init__('min_jerk_pose_planner_py')
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
        self.Pose, self.Time = [], []
        self.Pose_tool_all = np.zeros(0)

        self.force_inside_flag = False
        self.force_control_flag = False
        self.move_hori_flag = False
        self.i = 0
        self.N = 0
        self.n_now, self.m_now = 0, 0

        self.window_size = 20
        self.buffer = np.zeros((self.window_size, 3), dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.count = 0
        self.pos = 0

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
        if self.initial is None: ##确定任务规划的起点
            self.initial = msg
            self.cmd.position.x = self.initial.position.x
            self.cmd.position.y = self.initial.position.y
            self.cmd.position.z = self.initial.position.z
            self.cmd.orientation = self.initial.orientation
            self.get_logger().info('Initial pose received.')
        else:
            # 如果已有初始与目标 且 未启动轨迹
            if self.start_time is None:
                if any( v is None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
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
                    self.current = msg
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
            return -99.0 # 转换失败时返回-99

    def As_planner(self):
        if self.initial is not None and self.target is not None:
            M = Motion_planning(dx=0.001, dy=0.001, dz=0.001)
            start = np.array([self.initial.position.x, self.initial.position.y, self.initial.position.z])
            end = np.array([self.target.position.x, self.target.position.y, self.target.position.z])
            # Points_recover = M.path_searching(start=start, end=end)
            self.total_time = 0.0
            Points_recover = []
            Points_recover.append(start)
            q0 = np.array([self.initial.orientation.x,
                           self.initial.orientation.y,
                           self.initial.orientation.z,
                           self.initial.orientation.w])
            qd = q0
            if self.N == 0:  ##
                Points_recover.append(start + np.array([-0.160, -0.0, 0.0]))
                Points_recover.append(start + np.array([-0.180, 0.0, 0.0]))
                # Points_recover.append(np.array([673.26, 23.83, 350]) / 1000.0)
                # Points_recover.append(np.array([673.26, 23.83, 349]) / 1000.0)
                qd = np.array([0.0,
                               1.0,
                               0.0,
                               0.0])
            if self.N == 1:
                Points_recover.append(start + np.array([0.0, -0.120, 0.0]))
                Points_recover.append(start + np.array([0.00, -0.140, 0.0]))
                qd = q0
            # if self.N == 2:
            #     Points_recover.append(start + np.array([-0.0, 0.0, 0.01]))
            #     Points_recover.append(start + np.array([-0.0, 0.0, -0.0]))
            #     qd = q0
            if self.N > 2:
                self.get_logger().info('End the trajectory!')
            for j in range(len(Points_recover) - 1):
                self.total_time += max(np.linalg.norm(Points_recover[j + 1] - Points_recover[j]) / 0.006,
                                       abs(np.arccos(np.clip(np.dot(q0, qd), -1.0, 1.0)) * 2.0)/0.055)

            if Points_recover is not None:
                print(f'current_pose = {self.current}')
                self.X, self.Y, self.Z, self.Q = M.path_smoothing(Path_points=Points_recover,q0=q0, qf=qd, t_final=self.total_time,
                                                          freq=self.freq)  ##轨迹使用二次B样条曲线进行平滑处理
                self.get_logger().info(f"Q.shape={self.Q.shape}")
                self.get_logger().info("path_points get !")
            else:
                self.get_logger().info("the path is not found!!")

    def publish_at_time(self):
        self.cmd = self.current ##发布量的原始值为当前位置
        current_time = self.get_clock().now().to_msg().sec + \
                       self.get_clock().now().to_msg().nanosec * 1e-9
        ##进行触觉反馈探索
        if all( v is not None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
            self.F_x.append(f_x)
            self.F_y.append(f_y)
            self.F_z.append(f_z)
            self.R_x.append(r_x), self.R_y.append(r_y), self.R_z.append(r_z)
            self.Pose.append(np.array(
                            [self.current.position.x, self.current.position.y, self.current.position.z]))
            self.Time.append(current_time)

            ##计算合力矢量
            f_vec = np.array([f_x, f_y, f_z])

            ##计算偏移量(刚度越大，偏移量应该越小)
            delta_z = np.sign(f_z - 2) * min(0.0035, 0.0035 * abs(f_z - 2)) if abs(f_z)>0.3 else 0.0
            delta_x = np.sign(f_x - 0.05) * min(0.005, 0.005 * abs(f_x - 0.05)) if abs(f_x)>0.1 else 0.0
            delta_y = np.sign(f_y - 0.1) * min(0.005, 0.005 * abs(f_y - 0.1)) if abs(f_y)>0.2 else 0.0

            if self.force_control_flag: ##接触力调整阶段，保持一个恒定的接触力（目前为z方向恒定，后续扩展到其他方向）
                self.cmd.position.x += delta_x / self.freq
                self.cmd.position.y += delta_y / self.freq
                self.cmd.position.z += delta_z / self.freq

                ##合力在运动方向上的分量
                f_projec = (f_vec @ self.v_vec) / (self.v_vec @ self.v_vec) * self.v_vec
                if self.count < self.window_size: ##计算最近若干个值的均值
                    self.sum += f_projec
                    self.buffer[self.pos] = f_projec
                    self.count += 1
                    self.pos = (self.pos + 1) % self.window_size
                else:
                    self.sum += f_projec - self.buffer[self.pos]
                    self.buffer[self.pos] = f_projec
                    self.pos = (self.pos + 1) % self.window_size
                    f_projec_mean = self.sum / self.count

                    if 1.8 < np.linalg.norm(f_projec_mean) < 2.2:  ##分量的大小最近若干个值均在范围内，则认为接触力调整完成。
                        self.force_control_flag = False
                        self.get_logger().info('Contact force within threshold!')
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
                        if (np.linalg.norm(current_pose - last_pose) < 0.002 and
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

                        if np.linalg.norm(f_projec) > 2 and (f_vec @ self.v_vec) < 0:
                            self.force_control_flag = True  ##接触力超出阈值，进入接触力调整的阶段
                            ## 重置以接收新目标（但先不接收初始位置）
                            self.get_logger().info('Terminate and change contact force!')
                            self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
                            self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
                            self.total_time = 0.0
                            self.n_now, self.m_now = 0, 0
                            self.X, self.Y, self.Z, self.Q = None, None, None, None
                            self.force_inside_flag = False
                        else:
                            if self.n_now < self.X.shape[0]:  ##轨迹还未执行完，继续执行轨迹
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
                                        f'v_vec= {self.v_vec}, f_projec= {f_projec}, n_now= {self.n_now}/{self.X.shape[0]}')
                                    print(f'current_pose = {self.current.orientation}')
                            else:
                                self.cmd.position.x += delta_x / self.freq
                                self.cmd.position.y += delta_y / self.freq
                                self.cmd.position.z += delta_z / self.freq
                    else:
                        ## 重置以接收新目标
                        self.get_logger().info('Trajectory complete.')
                        self.initial = None
                        self.total_time = 0.0
                        self.n_now, self.m_now = 0, 0
                        self.X, self.Y, self.Z, self.Q = None, None, None, None
                        self.force_inside_flag = False

                else:
                    if self.N < 2:
                        self.As_planner()
                    else:
                        self.get_logger().info('The Planning Process has ended!')
                    self.N += 1
            self.pose_pub.publish(self.cmd)

            # if self.X is not None:
            #     now = self.get_clock().now().to_msg().sec + \
            #           self.get_clock().now().to_msg().nanosec * 1e-9
            #     t = now - self.start_time
            #     if t >= self.total_time:
            #         t = self.total_time
            #         self.get_logger().info('Trajectory complete.')
            #         ## 重置以接收新目标
            #         # self.start_time = None
            #         self.initial = None
            #         self.total_time = 0.0
            #         self.X, self.Y, self.Z = None, None, None
            #         self.force_inside_flag = False
            #     else:
            #         n_now = int(t * self.freq)
            #         if n_now >= self.X.shape[0]:
            #             n_now = self.X.shape[0]-1 ##避免溢出
            #         if f_x < 3 and f_y < 3 and f_z < 3: ##需要进一步细化设定不同的阈值，防止刚判断规划完就溢出
            #             self.cmd.position.x = self.X[n_now] - delta_x / self.freq
            #             self.cmd.position.y = self.Y[n_now] - delta_y / self.freq
            #             self.cmd.position.z = self.Z[n_now] - delta_z / self.freq
            #             # orientation 可插值或保持初始
            #             self.cmd.orientation = self.initial.orientation
            #         else:
            #             self.get_logger().info('Terminate and reasoning!')
            #             ## 重置以接收新目标
            #             # self.start_time = None
            #             self.initial = None
            #             self.total_time = 0.0
            #             self.X, self.Y, self.Z = None, None, None
            #             self.force_inside_flag = False
            #         self.pose_pub.publish(self.cmd)
            # else:
            #     if self.N < 2:
            #         self.As_planner()
            #         self.start_time = self.get_clock().now().to_msg().sec + \
            #                           self.get_clock().now().to_msg().nanosec * 1e-9
            #     else:
            #         self.get_logger().info('The Planning Process has ended!')
            #     self.N += 1

            # if self.i % 100 == 0:
            #         self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')
            #         self.get_logger().info(f'F_x= {f_x}, F_y= {f_y}, F_z= {f_z}')
            #         self.get_logger().info(f'delta_x= {delta_x}, delta_y= {delta_y}, delta_z= {delta_z}')

            # delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            # if -10 < self.Fr_r[0,2] < -8 and -10 < self.Fr_l[0,2] < -8 and self.force_inside_flag == False:
            #     delta_z = np.sign(f_z - 0.1) * min(0.005, 0.005 * abs(f_z - 0.1))
            #     delta_x = np.sign(f_x - 0.05) * min(0.005, 0.005 * abs(f_x - 0.05))
            #     delta_y = np.sign(f_y - 0.05) * min(0.005, 0.005 * abs(f_y - 0.05))
            #     self.force_inside_flag = True
            # elif -12 < self.Fr_r[0,2] < -6 and -12 < self.Fr_l[0,2] < -6 and self.force_inside_flag == True:
            #     delta_z = np.sign(f_z - 0.1) * min(0.005, 0.005 * abs(f_z - 0.1))
            #     delta_x = np.sign(f_x - 0.05) * min(0.005, 0.005 * abs(f_x - 0.05))
            #     delta_y = np.sign(f_y - 0.05) * min(0.005, 0.005 * abs(f_y - 0.05))
            # else:
            #     delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            #     self.force_inside_flag = False
            # self.cmd.position.z += delta_z / self.freq
            # self.cmd.position.x += delta_x / self.freq
            # self.cmd.position.y += delta_y / self.freq

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
        np.savez(r'Force_data_2508301147.npz', Force_left=node.Force_left, Force_right=node.Force_right,
                 Matrix_left=node.Matrix_left, Matrix_right=node.Matrix_right,
                 F_x=node.F_x, F_y=node.F_y, F_z=node.F_z, R_x=node.R_x, R_y=node.R_y, R_z=node.R_z,
                 Pose=node.Pose, Time=node.Time)
        print('save Data!')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
