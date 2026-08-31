import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

"""
本代码将机械臂末端位置控制的C++代码转化成了python代码。
"""

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

        # 发布器和订阅器
        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 1)
        self.pose_sub = self.create_subscription(
            Pose,
            '/lbr/state/pose',
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
        self.pose_pub.publish(cartesian_pose_cmd)

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

