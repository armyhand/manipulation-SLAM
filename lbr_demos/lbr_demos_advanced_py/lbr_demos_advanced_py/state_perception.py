import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from .motion_planning import Motion_planning
import ros2_numpy as rnp
from tutorial_interfaces.msg import Cloud, Array3
from .tac3d_test import gripper_force_flow, gripper_force_direct, contact_point_estimate, select_top_n, estimate_rigid_transform_kabsch
import pickle

import symforce
symforce.set_epsilon_to_invalid()

from symforce import geo
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.optimizer import OptimizerParams

"""
实现机器人末端、工具端的感知（定位）与工具和环境接触状态的判断（建图），并传输接触区域信息。
"""

class ContactSlamPerceptor(Node):
    def __init__(self):
        super().__init__('contact_slam_perceptor_py')
        self.Pose_tool_all = []
        self.diff_points = []
        self.row_index = None
        self.refer_points = []
        self.world_to_gripper_translation = None
        self.gripper_to_sensor_translation = None
        self.sensor_to_tool_translation_left = None
        self.sensor_to_tool_translation_right = None
        self.sensor_sense_translation_left = None
        self.sensor_sense_translation_right = None
        self.sensor_to_tool_permant = None
        self.gripper_pose_prior = None
        self.initial = None
        self.total_time = None
        self.sampling_time = 0.01
        self.freq = 100
        self.start_time = None
        self.X, self.Y, self.Z = None, None, None

        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 1)
        self.pose_sub = self.create_subscription(Pose, '/lbr/state/pose',
                                                 self.on_pose, 1)

        self.P_l = None  # 左手Tac3D marker点位置分布数据
        self.P_r = None  # 右手Tac3D marker点位置分布数据
        self.D_l = None # 左手marker点位移分布数据
        self.D_r = None # 右手marker点位移分布数据
        self.F_l = None # 左手力分布
        self.F_r = None # 右手力分布
        self.Fr_l = None # 左手合力数据
        self.Fr_r = None # 右手合力数据
        self.Mr_l = None  # 左手合力数据
        self.Mr_r = None  # 右手合力数据

        self.m = 0
        self.Force_left = []
        self.Force_right = []
        self.Matrix_left = []
        self.Matrix_right = []
        self.F_x, self.F_y, self.F_z, self.R_x, self.R_y, self.R_z = [], [], [], [], [], []
        self.F_x_1 = np.zeros(0)
        self.F_y_1 = np.zeros(0)
        self.F_z_1 = np.zeros(0)
        self.f_x_mean = np.zeros(0)
        self.f_y_mean = np.zeros(0)
        self.f_z_mean = np.zeros(0)
        self.G, self.Ll, self.Lr = [], [], []
        self.Rotation, self.Translation = [], []
        self.R_left = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ]) ## 将点的坐标从传感器坐标系转换到夹爪坐标系
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
        self.i = 1
        self.W = 10

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
            Array3, '/resultant_force_l', self.on_Fr_l, 10)
        self.Fr_r_sub = self.create_subscription(
            Array3, '/resultant_force_r', self.on_Fr_r, 10)
        self.Mr_l_sub = self.create_subscription(
            Array3, '/resultant_moment_l', self.on_Mr_l, 10)
        self.Mr_r_sub = self.create_subscription(
            Array3, '/resultant_moment_r', self.on_Mr_r, 10)
        
        # # 可选：从话题接收目标 Pose
        # self.target_sub = self.create_subscription(Pose, 'target/pose',
        #                                            self.on_target, 1)

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
        # self.get_logger().info(f'F_l array shape: {xyz.shape}')

    def on_F_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_r = xyz
        # self.get_logger().info(f'F_r array shape: {xyz.shape}')

    def on_Fr_l(self, msg: Cloud):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_l = np.array([x_val, y_val, z_val]).reshape(1,3)
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

            self.Fr_r = np.array([x_val, y_val, z_val]).reshape(1,3)
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

            self.Mr_l = np.array([x_val, y_val, z_val]).reshape(1,3)
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

            self.Mr_r = np.array([x_val, y_val, z_val]).reshape(1,3)
            self.Matrix_right.append(self.Mr_r)
            # self.get_logger().info(f'Mr_r: {self.Mr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')

    def on_pose(self, msg):
        if any(v is None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            self.get_logger().info('Waiting for tac3d information.')
        else:
            if -15 < self.Fr_r[0, 2] < -5 and -15 < self.Fr_l[0, 2] < -5:
                if len(self.refer_points) == 0:
                    self.row_index, _ = select_top_n(self.D_l, N=10, by_abs=True)
                    refer_point_left = self.P_l[self.row_index, :]
                    refer_point_left = np.dot(self.R_left, refer_point_left.T).T
                    refer_point_right = self.P_r[self.row_index, :]
                    refer_point_right = np.dot(self.R_right, refer_point_right.T).T
                    self.refer_points.append(refer_point_left)
                    self.refer_points.append(refer_point_right)
                else:
                    # 每一时刻夹爪的位置先验，由机械臂解算传回
                    self.gripper_pose_prior = geo.V3(msg.position.x*1000, msg.position.y*1000, msg.position.z*1000)
                    if all(v is not None for v in (self.P_l, self.P_r, self.D_l, self.D_r)):
                        self.sensor_to_tool_permant = geo.M44(1, 0, 0, 0,
                                                              0, 1, 0, 0,
                                                              0, 0, 1, -38.0,
                                                              0, 0, 0, 1)  # 工具坐标系相对于传感器坐标系的转换先验（定值，需事先标定）
                        diff_point_left = self.P_l[self.row_index, :]
                        diff_point_left = np.dot(self.R_left, diff_point_left.T).T
                        diff_point_right = self.P_r[self.row_index, :]
                        diff_point_right = np.dot(self.R_right, diff_point_right.T).T
                        # self.diff_points.append(diff_point_left)
                        # self.diff_points.append(diff_point_right)

                        R, T = estimate_rigid_transform_kabsch(self.refer_points[0], diff_point_left)
                        self.Rotation.append(R)
                        self.Translation.append(T)
                        self.sensor_sense_translation_left = geo.M44(R[0, 0], R[0, 1], R[0, 2], T[0],
                                                                R[1, 0], R[1, 1], R[1, 2], T[1],
                                                                R[2, 0], R[2, 1], R[2, 2], T[2],
                                                                0, 0, 0, 1)  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
                        self.sensor_to_tool_translation_left = self.sensor_sense_translation_left * self.sensor_to_tool_permant

                        R, T = estimate_rigid_transform_kabsch(self.refer_points[1], diff_point_right)
                        self.Rotation.append(R)
                        self.Translation.append(T)
                        self.sensor_sense_translation_right = geo.M44(R[0, 0], R[0, 1], R[0, 2], T[0],
                                                                     R[1, 0], R[1, 1], R[1, 2], T[1],
                                                                     R[2, 0], R[2, 1], R[2, 2], T[2],
                                                                     0, 0, 0, 1)  # 根据触觉感知解算出的传感器坐标系相对于夹爪坐标系的转换（变值）
                        self.sensor_to_tool_translation_right = self.sensor_sense_translation_right * self.sensor_to_tool_permant

                        self.gripper_to_sensor_translation = geo.M44(1, 0, 0, 0,
                                                                     0, 1, 0, 0,
                                                                     0, 0, 1, -200,
                                                                     0, 0, 0, 1)  # 传感器坐标系相对于夹爪（机械臂末端）坐标系的转换（定值）
                        self.world_to_gripper_translation = geo.M44(1, 0, 0, msg.position.x*1000,
                                                                    0, 1, 0, msg.position.y*1000,
                                                                    0, 0, 1, msg.position.z*1000,
                                                                    0, 0, 0, 1)

                        if self.m < 20:
                            f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l,
                                                                                self.Mr_r)
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
                            self.contact_optimizer()
            else:
                self.get_logger().info(f'Fr_r= {self.Fr_r}, Fr_l= {self.Fr_l}')

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

    def contact_optimizer(self):
        ##夹爪先验位置约束
        if self.i == 1:
            self.values[f"g{self.i}"] = geo.V3(0.0, 0.0, 0.0)
        else:
            self.values[f"g{self.i}"] = self.values[f"g{self.i-1}"]
        def gripper_pose_error(gi: geo.V3) -> geo.V3:
            return gi - self.gripper_pose_prior
        self.factors.append(Factor(keys=[f"g{self.i}"], residual=gripper_pose_error))

        ##工具位置约束
        if self.i == 1:
            self.values[f"ll{self.i}"] = geo.V3(0.0, 0.0, 0.0)
            self.values[f"lr{self.i}"] = geo.V3(0.0, 0.0, 0.0)
        else:
            self.values[f"ll{self.i}"] = self.values[f"ll{self.i - 1}"]
            self.values[f"lr{self.i}"] = self.values[f"lr{self.i - 1}"]
        def tool_pose_error_left(ll: geo.V3) -> geo.V3:
            return (self.world_to_gripper_translation * self.gripper_to_sensor_translation * self.sensor_to_tool_translation_left * geo.V4(0, 0, 0, 1))[0:3] - ll
        self.factors.append(Factor(keys=[f"ll{self.i}"], residual=tool_pose_error_left))

        def tool_pose_error_right(lr: geo.V3) -> geo.V3:
            return (self.world_to_gripper_translation * self.gripper_to_sensor_translation * self.sensor_to_tool_translation_right * geo.V4(0, 0, 0, 1))[0:3] - lr
        self.factors.append(Factor(keys=[f"lr{self.i}"], residual=tool_pose_error_right))

        # ##滑动窗口，只取最近N个因子进行优化
        # if self.i <= self.W:
        #     self.curr_optimized_values = self.values.copy()
        #     self.curr_optimized_factors = self.factors.copy()
        # else:
        #     for j in range(self.i - self.W, self.i):
        #         self.curr_optimized_values[f"g{j}"] = self.values[f"g{j}"]
        #         self.curr_optimized_values[f"l{j}"] = self.values[f"l{j}"]



        # ##触觉相对位置感知
        if all( v is not None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            f_x, f_y, f_z, r_x, r_y, r_z = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
            f_x = f_x - self.f_x_mean
            f_y = f_y - self.f_y_mean
            f_z = f_z - self.f_z_mean
            self.F_x.append(f_x)
            self.F_y.append(f_y)
            self.F_z.append(f_z)
            self.R_x.append(r_x), self.R_y.append(r_y), self.R_z.append(r_z)

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
        g_curr = self.values[f"g{self.i}"]
        l_curr_l = self.values[f"ll{self.i}"]
        l_curr_r = self.values[f"lr{self.i}"]
        if self.i % 10 == 0:
            self.get_logger().info(f"t={self.i}, g={g_curr}, ll={l_curr_l}, lr={l_curr_r}")
            self.get_logger().info(f"f_x={f_x}, f_y={f_y}, f_z={f_z}")
        self.G.append(np.array(g_curr))
        self.Ll.append(np.array(l_curr_l))
        self.Lr.append(np.array(l_curr_r))
        
        # self.Pose_tool_all.append(np.array(l_curr_r) / 1000)

        self.i += 1

def to_numpy_float(arr_list):
    return np.array(arr_list, dtype=np.float64)

def main(args=None):
    rclpy.init(args=args)
    node = ContactSlamPerceptor()

    # 设置拉杆：假定先设置 total_time，可通过 launch 参数或 Node 参数改动
    node.total_time = 8.0  # 例如 5 秒完成轨迹
    node.target = Pose()
    node.target.position.x = 0.0
    node.target.position.y = 0.0
    node.target.position.z = 0.0

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 保存 Values 对象到文件
        np.savez(r'State_260109.npz', Force_left=node.Force_left, Force_right=node.Force_right,
                 Matrix_left=node.Matrix_left, Matrix_right=node.Matrix_right,
                 F_x=node.F_x, F_y=node.F_y, F_z=node.F_z, R_x=node.R_x, R_y=node.R_y, R_z=node.R_z,
                 G=node.G, Ll=node.Ll, Lr=node.Lr, Rotation=node.Rotation, Translation=node.Translation, Pose_tool=node.Pose_tool_all)
        # with open("values_2509121804.pkl", "wb") as f:
        #     pickle.dump(node.values, f)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
