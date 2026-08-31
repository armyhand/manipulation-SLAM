-问题描述：以motion_planning_5.py作为源代码，场景中存在障碍物obstacle_true=np.array([[0,0], [0,-60],[20,-60],[20,-20],[80,-20],[80,-60],[100,-60],[100,-20],[160,-20],[160,-60],[180,-60],[180,0]])，我们的目标是通过movable_objects与obstacle_true的接触来检测、确定场景中obstacle_true真实的轮廓，也即将各个顶点的位置确定。

-已知条件：movable_objects的准确轮廓（相对自身参考点坐标系），使用源代码中的数据；obstacle_ref的轮廓坐标（也是相对自身参考点坐标系），obstacle_ref=0.7*obstacle_true；movable_objects的参考点的准确位置。轮廓坐标均为逆时针顺序。

-约束条件：obstacle_true的不相邻的边之间无相交；obstacle_true与obstacle_ref对应的边对应的方向矢量完全相同，也即obstacle_true相对obstacle_ref的边只会沿方向矢量伸缩，而不会改变方向；obstacle_true是一个闭合的多边形；movable_objects与obstacle_true的所有边均可发生碰撞，且两者无相交。

-策略概况：遵循原代码中的4方向离散运动生成方法，每次发生碰撞后生成可能的接触对（contact pair），再根据此时接触情况和约束条件优化obstacle_true各顶点的坐标，然后生成探索策略，逐步缩小contact pair的数目，直至只剩下唯一一对contact pair。探索策略的选择可以参考源代码的方法，也可以采用新的方法。

-其他要求：生成的代码和附件等放入新的文件夹中，展示每一步探索的结果。希望AI能够根据结果自动调整方法，如果探索不成功就更换方法，直至选择出动作次数最少、探索效率最高的动作策略。