import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from .motion_planning import Motion_planning

"""
本代码实现了使用A*算法进行轨迹规划和追踪
"""

class MinimumJerkPosePlanner(Node):
    def __init__(self):
        super().__init__('min_jerk_pose_planner_py')
        self.Q = None
        self.initial = None
        self.target = None
        self.total_time = 0.0
        self.sampling_time = 0.01
        self.freq = 100
        self.start_time = None
        self.X, self.Y, self.Z = None, None, None

        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 1)
        self.pose_sub = self.create_subscription(Pose, '/lbr/state/pose',
                                                 self.on_pose, 1)
        # # 可选：从话题接收目标 Pose
        # self.target_sub = self.create_subscription(Pose, 'target/pose',
        #                                            self.on_target, 1)

    def on_pose(self, msg):
        if self.initial is None:
            self.initial = msg
            self.get_logger().info('Initial pose received.')
        else:
            # 如果已有初始与目标 && 未启动轨迹
            if self.start_time is None:
                self.start_time = self.get_clock().now().to_msg().sec + \
                                   self.get_clock().now().to_msg().nanosec*1e-9
                self.get_logger().info('Starting minimum jerk trajectory.')

            # 如果轨迹已经启动
            if self.start_time is not None:
                self.publish_at_time()

    # def on_target(self, msg: Pose):
    #     self.target = msg
    #     self.get_logger().info('Target pose received.')

    def As_planner(self):
        if self.initial is not None and self.target is not None:
            M = Motion_planning(dx=0.05, dy=0.05, dz=0.05)
            start = np.array([self.initial.position.x, self.initial.position.y, self.initial.position.z])
            end = np.array([self.target.position.x, self.target.position.y, self.target.position.z])
            q0 = np.array([self.initial.orientation.x,
                  self.initial.orientation.y,
                  self.initial.orientation.z,
                  self.initial.orientation.w])
            qd = np.array([0.0,
                           1.0,
                           0.0,
                           0.0])
            # Points_recover = M.path_searching(start=start, end=end)
            Points_recover = []
            Points_recover.append(start), Points_recover.append(start+np.array([0.0, -0.02, 0.0]))
            Points_recover.append(start+np.array([0.0, -0.02, 0.0]))

            for j in range(len(Points_recover)-1):
                self.total_time += np.linalg.norm(Points_recover[j+1]-Points_recover[j]) / 0.005

            if Points_recover is not None:

                self.X, self.Y, self.Z, self.Q = M.path_smoothing(Path_points=Points_recover, q0=q0, qf=qd, t_final=self.total_time, freq=self.freq)  ##轨迹使用二次B样条曲线进行平滑处理
                self.get_logger().info("path_points get !")
            else:
                self.get_logger().info("the path is not found!!")

    def publish_at_time(self):
        cmd = Pose()

        # ##发布minimum-jerk得到的期望目标点
        # now = self.get_clock().now().to_msg().sec + \
        #       self.get_clock().now().to_msg().nanosec * 1e-9
        # t = now - self.start_time
        # if t > self.total_time:
        #     t = self.total_time
        # def mj_pos(p0, pf):
        #     tau = t / self.total_time
        #     return p0 + (pf - p0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
        #
        # cmd.position.x = mj_pos(self.initial.position.x,
        #                         self.target.position.x)
        # cmd.position.y = mj_pos(self.initial.position.y,
        #                         self.target.position.y)
        # cmd.position.z = mj_pos(self.initial.position.z,
        #                         self.target.position.z)
        # # orientation 可插值或保持初始
        # cmd.orientation = self.initial.orientation
        #
        # self.pose_pub.publish(cmd)

        ##发布A_star得到的期望目标点
        if self.X is not None:
            now = self.get_clock().now().to_msg().sec + \
                  self.get_clock().now().to_msg().nanosec * 1e-9
            t = now - self.start_time
            if t >= self.total_time:
                t = self.total_time
                self.get_logger().info('Trajectory complete.')
                # # 重置以接收新目标
                Points_recover = []
                self.total_time = 0.0
                # self.start_time = None
            else:
                n_now = int(t * self.freq)
                cmd.position.x = self.X[n_now]
                cmd.position.y = self.Y[n_now]
                cmd.position.z = self.Z[n_now]
                # orientation 可插值或保持初始
                quat = self.Q[n_now]
                cmd.orientation.x = float(quat[0])
                cmd.orientation.y = float(quat[1])
                cmd.orientation.z = float(quat[2])
                cmd.orientation.w = float(quat[3])

            self.pose_pub.publish(cmd)
        else:
            self.As_planner()

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
