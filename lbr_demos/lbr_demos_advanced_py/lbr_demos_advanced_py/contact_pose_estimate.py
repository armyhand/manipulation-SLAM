import numpy as np
import rclpy
from rclpy.node import Node
from lbr_fri_idl.msg import LBRState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Pose, Vector3
import tf2_ros
from scipy.spatial.transform import Rotation as R
from tf2_msgs.msg import TFMessage
from scipy.optimize import lsq_linear, minimize
from tutorial_interfaces.msg import Cloud
from .tac3d_test import gripper_force_direct, select_top_n, estimate_rigid_transform_kabsch

"""
本代码的功能是实现提取接触力矩和关节信息。
20260408: X方向的力检测不准确；Y方向的力漂移较大。需要对力和力矩信息进行校准。
260414：经过校准，X方向的机器人本体感知力误差最大，误差在90%以上；Y方向的力感知误差次之，在50%左右；Z方向的力感知误差最小，在15%左右。
"""

def objective(p_c, A, y, f_base, tau_ee, p_ee):

    # ===== 1. 关节力矩误差 =====
    term1 = A @ p_c - y
    loss1 = np.dot(term1, term1)

    # ===== 2. 末端力矩约束 =====
    moment = np.cross(p_c - p_ee, f_base)
    term2 = tau_ee - moment
    loss2 = np.dot(term2, term2)

    # ===== 权重 =====
    lambda2 = 1.0

    return loss1 + lambda2 * loss2

class ContactPoseEstimate(Node):
    def __init__(self):
        super().__init__('contact_pose_estimate')

        # 参数初始化
        self.is_init = False
        self.ext_torque = None  # 外部力矩
        self.ext_torque_init = None  # 初始外部力矩
        self.joint_position = None  # 关节位置
        self.transform_list = []  # TF链列表
        self.ext_force = None  # 外部力矢量
        self.ext_force_init = None  # 初始外部力矢量
        self.torque_ee = None  # 基于joint_ee的力矩
        self.torque_ee_init = None  # 基于joint_ee的初始力矩
        self.ee_position = None  # EE位置
        self.num = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.compute_contact)
        self.P_c_list = []  # 存储每次计算的接触点位置
        self.ext_force_list = []  # 存储每次计算的外部力矢量

        # 从pose_planning_node_8.py添加的数据变量
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

        self.F_init = None

        # 发布器和订阅器
        self.LBR_state_sub = self.create_subscription(
            LBRState,
            '/lbr/state',
            self.on_LBR_state,
            1
        )
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/lbr/force_torque_broadcaster/wrench',
            self.on_wrench,
            1
        )
        self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.pose_sub = self.create_subscription(Pose, '/lbr/state/pose',
                                                 self.on_pose, 1)

        # 从pose_planning_node_8.py添加的订阅器
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

    def skew(self, v):
        """叉乘矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def transform_to_matrix(self, trans):
        T = np.eye(4)

        t = trans.transform.translation
        q = trans.transform.rotation

        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]

        return T
    
    def transform_to_matrix_2(self, tf):
        T = np.eye(4)

        t = tf["transform"]["translation"]
        q = tf["transform"]["rotation"]

        T[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        T[:3, 3] = [t["x"], t["y"], t["z"]]

        return T
    
    def tf_callback(self, msg):
        transform_list = []
        for t in msg.transforms:
            # 只保留LBR链
            if "lbr_link" not in t.header.frame_id:
                continue

            transform_list.append({
                "header": {
                    "frame_id": t.header.frame_id
                },
                "child_frame_id": t.child_frame_id,
                "transform": {
                    "translation": {
                        "x": t.transform.translation.x,
                        "y": t.transform.translation.y,
                        "z": t.transform.translation.z,
                    },
                    "rotation": {
                        "x": t.transform.rotation.x,
                        "y": t.transform.rotation.y,
                        "z": t.transform.rotation.z,
                        "w": t.transform.rotation.w,
                    }
                }
            })
        # ⚠️ 需要排序！
        self.transform_list = sorted(
            transform_list,
            key=lambda x: x["child_frame_id"]
        )
    
    def on_LBR_state(self, msg: LBRState):
        # 提取外部力矩和关节位置信息
        if self.ext_torque_init is None:
            self.ext_torque_init = np.array(msg.external_torque).reshape(7)
        else:
            self.ext_torque = np.array(msg.external_torque).reshape(7) - self.ext_torque_init  # 机械臂有7个关节
        self.joint_position = np.array(msg.measured_joint_position).reshape(7)
    
    def on_wrench(self, msg: WrenchStamped):
        # 提取外部力矢量（基于joint_ee）
        if self.ext_force_init is None:
            self.ext_force_init = np.array([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z
            ])
            self.torque_ee_init = np.array([
                    msg.wrench.torque.x,
                    msg.wrench.torque.y,
                    msg.wrench.torque.z
                ])
        else:
            self.ext_force = np.array([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z
            ]) - self.ext_force_init
            self.torque_ee = np.array([
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z
            ]) - self.torque_ee_init
    
    def on_pose(self, msg: Pose):
        self.ee_position = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])
    
    # 从pose_planning_node_8.py添加的回调函数
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
        # self.get_logger().info(f'F_l array shape: {xyz.shape}')

    def on_F_r(self, msg: Cloud):
        xyz = np.vstack([msg.row1, msg.row2, msg.row3]).T  # shape (400,3)
        self.F_r = xyz
        # self.get_logger().info(f'F_r array shape: {xyz.shape}')

    def on_Fr_l(self, msg: Vector3):
        # 提取并清理XYZ数据
        if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
            x_val = self.clean_single_float(msg.x)
            y_val = self.clean_single_float(msg.y)
            z_val = self.clean_single_float(msg.z)

            self.Fr_l = np.array([x_val, y_val, z_val]).reshape(1, 3)
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
            # self.get_logger().info(f'Mr_r: {self.Mr_r}')
        else:
            self.get_logger().info(f'Don\'t Received left positions array shape')
    
    # ===== 获取 TF 链 =====
    def get_transform_list(self):

        frames = [
            ("lbr_link_0", "lbr_link_1"),
            ("lbr_link_1", "lbr_link_2"),
            ("lbr_link_2", "lbr_link_3"),
            ("lbr_link_3", "lbr_link_4"),
            ("lbr_link_4", "lbr_link_5"),
            ("lbr_link_5", "lbr_link_6"),
            ("lbr_link_6", "lbr_link_7"),
        ]

        transform_list = []

        for parent, child in frames:
            try:
                trans = self.tf_buffer.lookup_transform(
                    parent,
                    child,
                    rclpy.time.Time()
                )
                transform_list.append(trans)

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                return None

        return transform_list
    
    # ===== FK =====
    def compute_fk(self, transform_list):

        T = np.eye(4)

        p_list = []
        z_list = []

        for trans in transform_list:
            T_local = self.transform_to_matrix(trans)
            T = T @ T_local

            p = T[:3, 3]
            z = T[:3, 2]   # 关节轴（Z轴）

            p_list.append(p)
            z_list.append(z)

        return np.array(p_list), np.array(z_list), T
    
    def compute_fk_2(self, transform_list):

        T = np.eye(4)

        p_list = []
        z_list = []

        for trans in transform_list:
            T_local = self.transform_to_matrix_2(trans)
            T = T @ T_local

            p = T[:3, 3]
            z = T[:3, 2]   # 关节轴（Z轴）

            p_list.append(p)
            z_list.append(z)

        return np.array(p_list), np.array(z_list), T

    def compute_contact(self):
        if self.joint_position is None or self.ext_torque is None or self.ext_force is None or self.ee_position is None or self.torque_ee is None or self.Fr_r is None:
            return
        # ===== FK =====
        transform_list = self.get_transform_list()
        if transform_list is None:
            return
        p_list, z_list, T_ee = self.compute_fk(transform_list)
        # p_list_1, z_list_1, T_ee_1 = self.compute_fk_2(self.transform_list)  # 来自tf_callback的TF链

        # ===== 力转换：EE → BASE =====
        R_ee = T_ee[:3, :3]
        # f_base = R_ee @ self.ext_force
        torque_ee_base = R_ee @ self.torque_ee
        Fr_r_base = np.array([-self.Fr_r[0, 1], self.Fr_r[0, 2], -self.Fr_r[0, 0]])
        if all(v is not None for v in (self.Fr_r, self.Fr_l, self.Mr_r, self.Mr_l)):
            if self.F_init is None:
                f_x, f_y, f_z,_,_,_ = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
                self.F_init = np.array([f_x, f_y, f_z])
                f_base = np.zeros(3)
            else:
                f_x, f_y, f_z,_,_,_ = gripper_force_direct(self.Fr_l, self.Fr_r, self.Mr_l, self.Mr_r)
                f_base = np.array([f_x, f_y, f_z]) - self.F_init

        # ===== 构建 A, b =====
        A = []
        b = []

        f_skew = self.skew(f_base)

        for i in range(7):
            zi = z_list[i]
            pi = p_list[i]

            Ai = zi @ f_skew
            bi = zi @ (f_skew @ pi)

            A.append(Ai)
            b.append(bi)

        A = np.array(A)
        b = np.array(b)

        # ===== 最小二乘求解 =====
        # try:
        #     p_c = np.linalg.inv(A.T @ A + 1e-6*np.eye(3)) @ A.T @ (b - self.ext_torque)
        # except Exception as e:
        #     self.get_logger().warn(f"Solve failed: {e}")
        #     return
        
        # #======带约束求解=======#
        # y = b - self.ext_torque
        # # ===== 约束范围（你可以根据机器人实际尺寸改）=====
        # # 已知末端物体在末端坐标系下的包络范围（单位 mm）：
        # # 定义物体在末端坐标系下的最小/最大角点（局部包络盒）
        # envelope_min_local = np.array([-35.0, -48.0, -15.0])
        # envelope_max_local = np.array([35.0, 48.0, 350.0])
        # # 将局部包络的 8 个角点转换到基座系，然后按轴取 min/max，避免直接使用单个角点导致符号/顺序问题
        # corners_local = np.array([
        #     [envelope_min_local[0], envelope_min_local[1], envelope_min_local[2]],
        #     [envelope_min_local[0], envelope_min_local[1], envelope_max_local[2]],
        #     [envelope_min_local[0], envelope_max_local[1], envelope_min_local[2]],
        #     [envelope_min_local[0], envelope_max_local[1], envelope_max_local[2]],
        #     [envelope_max_local[0], envelope_min_local[1], envelope_min_local[2]],
        #     [envelope_max_local[0], envelope_min_local[1], envelope_max_local[2]],
        #     [envelope_max_local[0], envelope_max_local[1], envelope_min_local[2]],
        #     [envelope_max_local[0], envelope_max_local[1], envelope_max_local[2]],
        # ])
        # # 单位转换为米并旋转到基座系，再加上末端位置得到基座系坐标
        # corners_local_m = corners_local / 1000.0
        # corners_base = (R_ee @ corners_local_m.T).T + self.ee_position
        # p_min = corners_base.min(axis=0)
        # p_max = corners_base.max(axis=0)
        # lambda_reg = 1e-3
        # p_prior = self.ee_position + (R_ee @ np.array([0.0, 0.0, 200.0]))/1000  # 可设为末端附近
        # A_aug = np.vstack([A, np.sqrt(lambda_reg)*np.eye(3)])
        # y_aug = np.hstack([y, np.sqrt(lambda_reg)*p_prior])
        # try:
        #     res = lsq_linear(A_aug, y_aug, bounds=(p_min, p_max), method='trf')

        #     if res.success:
        #         p_c = res.x
        #     else:
        #         self.get_logger().warn("Optimization failed")
        #         return
        # except Exception as e:
        #     self.get_logger().warn(f"Optimization error: {e}")
        #     return

        #==========增加末端约束===================#
        y = b - self.ext_torque
        p_ee = T_ee[:3, 3]
        # p_prior: 初始猜测（可调整）
        p_prior = self.ee_position + (R_ee @ np.array([0.0, 0.0, 200.0]))/1000  # 可设为末端附近

        # 已知末端物体在末端坐标系下的包络范围（单位 mm）：
        # 定义物体在末端坐标系下的最小/最大角点（局部包络盒）
        envelope_min_local = np.array([-35.0, -48.0, -15.0])
        envelope_max_local = np.array([35.0, 48.0, 350.0])

        # 将局部包络的 8 个角点转换到基座系，然后按轴取 min/max，避免直接使用单个角点导致符号/顺序问题
        corners_local = np.array([
            [envelope_min_local[0], envelope_min_local[1], envelope_min_local[2]],
            [envelope_min_local[0], envelope_min_local[1], envelope_max_local[2]],
            [envelope_min_local[0], envelope_max_local[1], envelope_min_local[2]],
            [envelope_min_local[0], envelope_max_local[1], envelope_max_local[2]],
            [envelope_max_local[0], envelope_min_local[1], envelope_min_local[2]],
            [envelope_max_local[0], envelope_min_local[1], envelope_max_local[2]],
            [envelope_max_local[0], envelope_max_local[1], envelope_min_local[2]],
            [envelope_max_local[0], envelope_max_local[1], envelope_max_local[2]],
        ])

        # 单位转换为米并旋转到基座系，再加上末端位置得到基座系坐标
        corners_local_m = corners_local / 1000.0
        corners_base = (R_ee @ corners_local_m.T).T + self.ee_position

        p_min = corners_base.min(axis=0)
        p_max = corners_base.max(axis=0)
        bounds = [
                    (p_min[0], p_max[0]),
                    (p_min[1], p_max[1]),
                    (p_min[2], p_max[2]),
                ]
        res = minimize(
                    objective,
                    p_prior,  # 初始猜测
                    args=(A, y, f_base, torque_ee_base, self.ee_position),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
        if res.success:
            p_c = res.x
        else:
            self.get_logger().warn("Optimization failed")
            return

        
        if self.num % 10 == 0:  # 每10次打印一次
            # self.get_logger().info(f'z_list from callback: {z_list_1}, z_list from tf2ros: {z_list}')
            self.get_logger().info(f'External Force: {f_base}')
            self.get_logger().info(f'External Joint Torque: {self.ext_torque}')
            self.get_logger().info(f'ee_position: {self.ee_position}')
            # self.get_logger().info(f'bounds: {bounds}')
            if np.linalg.norm(f_base) > 0.5:  # 仅当力较大时打印估计结果
                self.get_logger().info(f'Estimated Contact Point (BASE): {p_c}')
                self.P_c_list.append(p_c)  # 存储估计的接触点位置
                self.ext_force_list.append(f_base)  # 存储对应的外部力
            else:
                self.get_logger().info(f'Estimated Contact Point (BASE): None')
                self.P_c_list.append(np.array([0, 0, 0]))  # 存储估计的接触点位置
                self.ext_force_list.append(f_base)  # 存储对应的外部力
            self.get_logger().info('***************************************')
        self.num += 1


def main(args=None):
    rclpy.init(args=args)
    node = ContactPoseEstimate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        np.savez('contact_estimate_data.npz', P_c=node.P_c_list, ext_force=node.ext_force_list)
        print('save Data!')
        node.destroy_node()
        rclpy.shutdown()
        

if __name__ == '__main__':
    main()