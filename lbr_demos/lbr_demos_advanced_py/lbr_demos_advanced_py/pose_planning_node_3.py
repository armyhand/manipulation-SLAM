import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R, Slerp

class PosePlanningNode(Node):

    def __init__(self):
        super().__init__('pose_planning_py')

        # 参数初始化
        self.initial_pose = None  # 第一帧接收到的位姿
        self.is_init = False

        self.amplitude = 0.05  # 弹幅 (m)
        self.frequency = 0.1   # 频率 (Hz)
        self.sampling_time = 0.01  # 发送命令采样时间 (s)
        self.phase = 0.0       # 初相位
        self.max_step = 0.1 # 最大步幅
        self.max_angle_step = 0.1  # (in radians)

        # 发布器和订阅器
        self.pose_pub = self.create_publisher(Pose, 'command/pose', 1)
        self.pose_sub = self.create_subscription(
            Pose,
            'state/pose',
            self.on_pose,
            1
        )

    def on_pose(self, msg: Pose):
        if not self.is_init:
            # 记录初始位姿
            self.initial_pose = msg
            self.is_init = True
            self.get_logger().info('Initial pose set.')
            return

        # 复制初始 pose 作为基准
        cartesian_pose_cmd = Pose()
        cartesian_pose_cmd.position.x = self.initial_pose.position.x
        cartesian_pose_cmd.position.y = self.initial_pose.position.y
        cartesian_pose_cmd.position.z = self.initial_pose.position.z

        cartesian_pose_cmd.orientation = msg.orientation  # 或者 initial_pose.orientation, 视需求

        # 更新相位并计算 x 方向上的正弦位移
        self.phase += 2 * 3.14 * self.frequency * self.sampling_time
        cartesian_pose_cmd.position.x += self.amplitude * np.sin(self.phase)

        # 发布命令位姿
        #self.pose_pub.publish(cartesian_pose_cmd)
        self.move_towards_pose(self.initial_pose, cartesian_pose_cmd)
        
    def move_towards_pose(self, current_pose, desired_pose):
        # Interpolate position
        position_diff = np.array([
            desired_pose.position.x - current_pose.position.x,
            desired_pose.position.y - current_pose.position.y,
            desired_pose.position.z - current_pose.position.z
        ])

        distance = np.linalg.norm(position_diff)
        if distance > self.max_step:
            step = position_diff / distance * self.max_step
        else:
            step = position_diff

        # Interpolate quaternion (using spherical linear interpolation)
        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])

        desired_quat = np.array([
            desired_pose.orientation.x,
            desired_pose.orientation.y,
            desired_pose.orientation.z,
            desired_pose.orientation.w
        ])

        # Calculate interpolation ratio
        angle_diff = np.arccos(np.clip(np.dot(current_quat, desired_quat), -1.0, 1.0)) * 2.0
        if angle_diff > self.max_angle_step:
            t = self.max_angle_step / angle_diff
        else:
            t = 1.0

        key_rots = R.from_quat([current_quat, desired_quat])
        slerp = Slerp([0, 1], key_rots)
        interpolated_quat = slerp(t).as_quat()

        intermediate_pose = Pose()
        intermediate_pose.position.x = current_pose.position.x + step[0]
        intermediate_pose.position.y = current_pose.position.y + step[1]
        intermediate_pose.position.z = current_pose.position.z + step[2]

        intermediate_pose.orientation.x = interpolated_quat[0]
        intermediate_pose.orientation.y = interpolated_quat[1]
        intermediate_pose.orientation.z = interpolated_quat[2]
        intermediate_pose.orientation.w = interpolated_quat[3]

        self.pose_pub.publish(intermediate_pose)

def main(args=None):
    rclpy.init(args=args)
    node = PosePlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

