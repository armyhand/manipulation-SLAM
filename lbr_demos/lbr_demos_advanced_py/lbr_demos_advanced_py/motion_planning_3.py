import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import splrep, splev
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import KDTree

"""
push block的探索与规划代码库
"""

class Motion_planning():
    def __init__(self, d, padding):
        self.y_max_socket = None
        self.y_min_socket = None
        self.x_max_socket = None
        self.x_min_socket = None
        self.Weight_free = None
        self.Pos_free = None
        self.normals_obstacles = None
        self.normals_objects = None
        self.Gain_information = None
        self.Gain_information_optimize = None
        self.Dis_var = None
        self.z_new = None
        self.y_new = None
        self.x_new = None
        self.Q = None
        self.t = None
        self.n = None
        self.y_max = 0
        self.x_max = 0
        self.y_min = 0
        self.x_min = 0
        self.obstacles = None
        self.scene = None
        self.d = d
        self.padding = padding
        self.actions = [np.array([1,0]), np.array([-1,0]),
                        np.array([0,1]), np.array([0,-1])]
        self.r = 7.0 ##重采样半径范围

    ## 原类中一些函数要继承
    def path_smoothing(self, Path_points, q0, qf, t_final, freq):
        self.n = int(t_final * freq)

        Path_points = np.array(Path_points)
        # 分别对每个坐标轴进行插值
        t = np.arange(len(Path_points))
        tck_x = splrep(t, Path_points[:, 0], k=2, s=0)
        tck_y = splrep(t, Path_points[:, 1], k=2, s=0)
        tck_z = splrep(t, Path_points[:, 2], k=2, s=0)
        # 生成插值后的坐标点
        t_new = np.linspace(0, len(Path_points) - 1, self.n)
        self.x_new = splev(t_new, tck_x)
        self.y_new = splev(t_new, tck_y)
        self.z_new = splev(t_new, tck_z)

        # 生成姿态的插值
        key_rots = R.from_quat([q0, qf])
        key_times = [0, self.n]
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, self.n, self.n)
        self.Q = slerp(times).as_quat()  # shape (N,4)


        return self.x_new, self.y_new, self.z_new, self.Q

    ##离散化环境和障碍物,物体-----------------------------------
    def discretize_scene(self, obstacles_1):
        """
        obstacles: list of np.array, 每个元素是(N,2)的数组，表示一个障碍物轮廓（逆时针散点）
        d: 步长 (mm)
        """
        self.obstacles = obstacles_1
        # 1. 计算边界矩形
        all_points = np.vstack(obstacles_1)
        self.x_min, self.y_min = np.min(all_points, axis=0) - np.array([self.padding[0], self.padding[2]])
        self.x_max, self.y_max = np.max(all_points, axis=0) + np.array([self.padding[1], self.padding[3]])


        # 2. 生成网格
        xs_1 = np.arange(self.x_min, self.x_max, self.d)
        ys_1 = np.arange(self.y_min, self.y_max, self.d)
        X, Y = np.meshgrid(xs_1, ys_1)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        # 3. 初始化场景
        self.scene = np.zeros(X.shape, dtype=np.int32)

        # 4. 对每个障碍物，判断点是否在内部
        for obs in obstacles_1:
            path = Path(obs)
            mask = path.contains_points(grid_points)
            mask = mask.reshape(X.shape)
            self.scene[mask] = 1  # 障碍物区域设为1

        return self.scene, xs_1, ys_1

    def pos_to_Map(self, pos): ##从坐标转换为离散地图的位置
        Map_pos = [int((pos[0] - self.x_min) / self.d), int((pos[1] - self.y_min) / self.d)]

        return Map_pos

    def add_movable_objects(self, objects, ref_point_1):
        off = np.asarray(ref_point_1, dtype=float)
        objects_pos = [poly + off for poly in objects]

        all_points = np.vstack(self.obstacles)
        self.x_min, self.y_min = np.min(all_points, axis=0) - np.array([self.padding[0], self.padding[2]])
        self.x_max, self.y_max = np.max(all_points, axis=0) + np.array([self.padding[1], self.padding[3]])
        # ##调试打印
        # print("x_min:", self.x_min, type(self.x_min))
        # print("x_max:", self.x_max, type(self.x_max))
        # print("d:", self.d, type(self.d))

        xs_2 = np.arange(self.x_min, self.x_max, self.d)
        ys_2 = np.arange(self.y_min, self.y_max, self.d)
        X, Y = np.meshgrid(xs_2, ys_2)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        mask = np.zeros(X.shape, dtype=bool)
        for obs in objects_pos:
            path = Path(obs)
            inside = path.contains_points(grid_points).reshape(X.shape)
            mask |= inside

        return mask, objects_pos

    ##产生粒子和权重，并更新权重---------------------------------------
    def generate_particles_in_polygon(self, polygon_all, N):
        """
        polygon: np.array (N,2)，表示封闭轮廓
        xs, ys: 网格坐标 (来自 discretize_scene)
        return: 粒子位置 np.array (M,2)，粒子权重 np.array (M,)
        """
        # 构造 shapely 多边形集合
        polys = [Polygon(p) for p in polygon_all]
        region = unary_union(polys)  # 取并集，得到完整区域

        # 得到边界框
        rx_min, ry_min, x_max, y_max = region.bounds

        # 估计网格分辨率（尽量接近 sqrt(N) × sqrt(N)）
        n_side = int(np.ceil(np.sqrt(N)))
        xs_3 = np.linspace(rx_min, x_max, n_side)
        ys_3 = np.linspace(ry_min, y_max, n_side)
        X, Y = np.meshgrid(xs_3, ys_3)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        # 过滤落在多边形区域内部的点
        inside_points = np.array([pt for pt in grid_points if region.contains(Point(pt))])

        # 如果点太多，均匀下采样
        if len(inside_points) > N:
            idx = np.linspace(0, len(inside_points) - 1, N, dtype=int)
            inside_points = inside_points[idx]
        elif len(inside_points) < N:
            # 粒子不足：增加网格分辨率
            factor = int(np.ceil(np.sqrt(N / len(inside_points))))
            xs_3 = np.linspace(rx_min, x_max, n_side * factor)
            ys_3 = np.linspace(ry_min, y_max, n_side * factor)
            X, Y = np.meshgrid(xs_3, ys_3)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T
            inside_points = np.array([pt for pt in grid_points if region.contains(Point(pt))])
            # 再次裁剪为 N 个
            if len(inside_points) > N:
                idx = np.linspace(0, len(inside_points) - 1, N, dtype=int)
                inside_points = inside_points[idx]
        particles_1 = inside_points.reshape(-1, 2)
        particles_1 = np.unique(particles_1, axis=0)
        N1 = len(particles_1)
        weights_1 = np.ones(N1) / N1  # 均匀初始化

        return particles_1, weights_1

    def find_local_maximum(self, particles_2, weights_2, radius=0.2, mode="max"):
        peaks_1 = []
        all_equal = np.all(weights_2 == weights_2[0])
        if all_equal and len(particles_2) > 10:
            return peaks_1
        else:
            # 粒子的权重大于weight_thre的都这值为局部最优
            w_thresh = 1.0 / (len(particles_2) * 2)
            mask = weights_2 >= w_thresh
            particles_maximum = particles_2[mask]
            weights_maximum = weights_2[mask]
            for i in range(len(particles_maximum)):
                ix, iy = particles_maximum[i]
                val = weights_maximum[i]
                peaks_1.append([np.array([ix, iy]), val])

            return peaks_1
        
    def plan_trajectory(self, start, goal, objects):
        """
        规划从起点到终点的无碰撞最短轨迹。
        参数:
            start (np.array): 起始位置 [x,y]
            goal (np.array): 目标位置 [x,y]
            obstacles (list of np.array): 每个元素为 (N,2) 的障碍物轮廓
            objects (list of np.array, optional): 可移动物体轮廓
        返回:
            action_seq (list of np.array): 动作序列，每个动作为长度为 d 的位移向量
            如果无法到达返回 None。
        """
        # 1. 离散化环境，得到栅格地图
        occ = self.scene.copy()  # 0: 空闲, 1: 障碍
        # 2. 将可移动物体也标为占用
        if objects is not None and len(objects) > 0:
            mask, obj_curr = self.add_movable_objects(objects, start)
            from shapely.geometry import Polygon
            for obj in obj_curr:
                poly_obj = Polygon(obj)
                area_obj = poly_obj.area
                if area_obj <= 0:
                    continue
                overlap_area = 0.0
                for obs in self.obstacles:
                    inter = poly_obj.intersection(Polygon(obs))
                    overlap_area += inter.area
                if overlap_area / area_obj > 0.001:
                    # 物体与障碍物重叠过多，不可行
                    print(f"Object at {start} overlaps with obstacles. No valid path.")
                    return [], []

        # 3. 将起点和终点转换为地图索引
        start_idx = tuple(self.pos_to_Map(start))
        goal_idx = tuple(self.pos_to_Map(goal))

        # 检查起点或终点是否在障碍内
        if (not (0 <= start_idx[0] < scene.shape[1] and 0 <= start_idx[1] < scene.shape[0])) or \
           (not (0 <= goal_idx[0] < scene.shape[1] and 0 <= goal_idx[1] < scene.shape[0])):
            print(f"Start {start} or goal {goal} is out of bounds. No valid path.")
            return [], []
        # if occ[start_idx[1], start_idx[0]] == 1 or occ[goal_idx[1], goal_idx[0]] == 1:
        #     print(f"Start {start} or goal {goal} is in an occupied cell. No valid path.")
        #     return [], []

        # 4. 广度优先搜索 (BFS) 寻找最短路径
        x_min, x_max, y_min, y_max = -26, 346, -26, 166
        action_seq = []
        distance_seq = []
        if x_min < start[0] < x_max:
            if y_min < start[1] < y_max:
                d1 = abs(y_min - start[1])
                a1 = np.array([0, -1])
                d2 = min(abs(x_min - start[0]), abs(x_max - start[0]))
                a2 = np.array([-1, 0]) if abs(x_min - start[0]) < abs(x_max - start[0]) else np.array([1, 0])
                d3 = abs(goal[1] - y_min)
                a3 = np.array([0, 1])
                d4 = abs(goal[0] - x_min) if np.array_equal(a2, np.array([-1, 0])) else abs(goal[0] - x_max)
                a4 = -a2
                action_seq = [a1, a2, a3, a4]
                distance_seq = [d1, d2, d3, d4]
            elif start[1] <= y_min:
                d1 = min(abs(x_min - start[0]), abs(x_max - start[0]))
                a1 = np.array([-1, 0]) if abs(x_min - start[0]) < abs(x_max - start[0]) else np.array([1, 0])
                d2 = abs(goal[1] - start[1])
                a2 = np.array([0, 1])
                d3 = abs(goal[0] - x_min) if np.array_equal(a1, np.array([-1, 0])) else abs(goal[0] - x_max)
                a3 = -a1
                action_seq = [a1, a2, a3]
                distance_seq = [d1, d2, d3]
            else:
                d1 = abs(goal[1] - start[1])
                a1 = np.array([0, 1]) if goal[1] > start[1] else np.array([0, -1])
                d2 = abs(goal[0] - start[0])
                a2 = np.array([1, 0]) if goal[0] > start[0] else np.array([-1, 0])
                action_seq = [a1, a2]
                distance_seq = [d1, d2]
        else:
            d1 = abs(goal[1] - start[1])
            a1 = np.array([0, 1]) if goal[1] > start[1] else np.array([0, -1])
            d2 = abs(goal[0] - start[0])
            a2 = np.array([1, 0]) if goal[0] > start[0] else np.array([-1, 0])
            action_seq = [a1, a2]
            distance_seq = [d1, d2]

        if len(action_seq) == 0:
            print(f"No path found from {start} to {goal}.")
            return [], []
        else:
            return action_seq, distance_seq
        
    def socket_curve(self, padding_1):
        all_points = np.vstack(self.obstacles)
        self.x_min_socket, self.y_min_socket = np.min(all_points, axis=0) - np.array([padding_1[0], padding_1[2]])
        self.x_max_socket, self.y_max_socket = np.max(all_points, axis=0) + np.array([padding_1[1], padding_1[3]])

        return np.array([self.x_min_socket, self.x_max_socket, self.y_min_socket, self.y_max_socket])


    def update_particles(self, particles_3, weights_3, polygons, shift_3=np.array([0, 0])):
        """
        输入:
            particles: (N,2) 粒子坐标
            weights: (N,) 对应粒子权重
            polygons: list，每个元素是 (M,2) 顶点数组（逆时针顺序的封闭多边形）
            shift: (dx, dy)，平移向量
            w_thresh: 权重阈值，小于此值的粒子会被剔除
        输出:
            new_particles: 平移+筛选后保留下来的粒子
            new_weights: 对应权重
        """
        # 1. 剔除低权重粒子
        w_thresh = 1.0 / (len(particles_3) * 2)
        mask = weights_3 >= w_thresh
        particles_3 = particles_3[mask]
        # weights = weights[mask]

        if len(particles_3) == 0:
            return np.zeros((0, 2)), np.zeros(0)

        # 2. 平移粒子
        particles_shifted = particles_3 + np.array(shift_3)

        # 3. 判断是否在任意多边形内
        inside_mask_total = np.zeros(len(particles_shifted), dtype=bool)
        for poly in polygons:
            path = Path(poly)
            inside_mask_total |= path.contains_points(particles_shifted)

        # 4. 保留在多边形内的粒子
        new_particles = particles_shifted[inside_mask_total]
        # 如果粒子数目过少，需要生成新的粒子
        if len(new_particles) < 30:
            # 随机角度
            theta = np.random.uniform(0, 2 * np.pi, size=4)
            # 随机半径 (平方根保证均匀分布在圆内)
            r=4
            radius = r * np.sqrt(np.random.uniform(0, 1, size=4))
            # 极坐标转直角坐标
            Points = np.zeros(0)
            for p in new_particles:
                x = p[0] + radius * np.cos(theta)
                y = p[1] + radius * np.sin(theta)
                points = np.column_stack((x, y))
                Points = np.append(Points, points)
                Points.reshape(-1, 2)
            new_particles = np.append(new_particles, Points).reshape(-1, 2)
            new_particles = np.unique(new_particles, axis=0)

        N = new_particles.shape[0]
        new_weights = np.ones(N) / N

        return new_particles, new_weights

    def likelihood(self, obs, pred, sigma=1.5):
        """ 高斯似然（独立同分布） """
        diff = float(obs) - float(pred)
        return math.exp(-0.5 * np.dot(diff, diff) / (sigma ** 2))

    ##筛选接触对---------------------------------------------------
    def n_vector(self, obstacles_4, type_1='stay'):
        """计算障碍物轮廓上每一点的内法线,可移动物体上每一点的外法线"""
        normals = []
        if type_1 == 'stay':
            for ci, contour in enumerate(obstacles_4):
                n = len(contour)
                # 保证闭合
                contour = np.vstack([contour, contour[0]])
                for i in range(n):
                    p1 = contour[i]
                    p2 = contour[i + 1]
                    t = p2 - p1  # 切向量
                    t = t / np.linalg.norm(t)  # 归一化
                    vector_in = np.array([-t[1], t[0]])
                    normals.append([p1, p2, vector_in, ci])
        elif type_1 == 'movable':
            for ci, contour in enumerate(obstacles_4):
                n = len(contour)
                # 保证闭合
                contour = np.vstack([contour, contour[0]])
                for i in range(n):
                    p1 = contour[i]
                    p2 = contour[i + 1]
                    t = p2 - p1  # 切向量
                    t = t / np.linalg.norm(t)  # 归一化
                    vector_out = np.array([t[1], -t[0]])
                    normals.append([p1, p2, vector_out, ci])

        return normals

    def select_pairs(self, action, objects):
        """
        这里的objects坐标一定要用相对ref_point的坐标
        """
        poten_con_obj_lines = []
        poten_con_obs_lines = []
        self.normals_objects = self.n_vector(objects, type_1='movable')
        self.normals_obstacles = self.n_vector(self.obstacles, type_1='movable')
        for n in self.normals_objects:
            normal_out = n[2]
            dot = np.dot(action, normal_out)
            if dot > 0.1:
                poten_con_obj_lines.append([np.array([n[0], n[1]]), n[3]])
        for n in self.normals_obstacles:
            normal_in = n[2]
            dot = np.dot(action, normal_in)
            if dot < -0.1:
                poten_con_obs_lines.append([np.array([n[0], n[1]]), n[3]])

        ##划分接触对,然后分析每个接触对对应的参考点的运动范围
        poten_con_pairs_1 = []
        for i in range(len(poten_con_obj_lines)):
            obj_line_1 = poten_con_obj_lines[i][0]
            c_obj = poten_con_obj_lines[i][1]
            obj_xmin, obj_ymin = np.min(obj_line_1, axis=0)
            obj_xmax, obj_ymax = np.max(obj_line_1, axis=0)
            for j in range(len(poten_con_obs_lines)):
                obs_line_1 = poten_con_obs_lines[j][0]
                c_obs = poten_con_obs_lines[j][1]
                obs_xmin, obs_ymin = np.min(obs_line_1, axis=0)
                obs_xmax, obs_ymax = np.max(obs_line_1, axis=0)

                ref_point_xmin = obs_xmin - obj_xmax
                ref_point_ymin = obs_ymin - obj_ymax
                ref_point_xmax = obs_xmax - obj_xmin
                ref_point_ymax = obs_ymax - obj_ymin

                range_point = np.array([[ref_point_xmin, ref_point_ymin],
                                        [ref_point_xmax, ref_point_ymax]])
                poten_con_pairs_1.append([obj_line_1, obs_line_1, range_point,
                                        np.array([c_obj,c_obs])])

        return poten_con_obj_lines, poten_con_obs_lines, poten_con_pairs_1

    ##根据接触状态预测触觉信号（z=0,1），选择运动方向-----------------------------------------------
    def obs_pred(self, action, Object, ref_point_3):
        def point_to_segment_distance(p, a, b):
            """
            计算点p到线段ab的最短距离
            """
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)  # 投影参数
            t = max(0, min(1, t))  # 限制在 [0,1] 内
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        objects_mask, object_pose = self.add_movable_objects(Object, ref_point_3)
        # offset = np.asarray(action * self.d, dtype=float)
        ##如果超出孔区域范围，直接返回-1。

        if ref_point_3[0] > self.x_max or ref_point_3[1] > self.y_max \
                or ref_point_3[0] < self.x_min or ref_point_3[1] < self.y_min:
            print(f"Out of the scope!!")
            z_pred = -1
            return z_pred, None, None
        else:
            # object_pose_next = [poly + offset for poly in object_pose]
            _, _, poten_con_pairs_2 = self.select_pairs(action, object_pose)
            # _,_,poten_con_pairs_next = self.select_pairs(action, object_pose_next)
            z_pred = 0
            obj_line_2, obs_line_2 = None, None

            ##判断当前潜在接触对的相交情况，不相交的话判断距离情况
            for con_pair in poten_con_pairs_2:
                obj_line_2, obs_line_2 = con_pair[0], con_pair[1]
                p1, p2 = obj_line_2
                p3, p4 = obs_line_2

                d1 = p2 - p1
                d2 = p4 - p3
                A = np.array([d1, -d2]).T
                b = p3 - p1
                det = np.linalg.det(A)

                if abs(det) > 1e-10:
                    t, u = np.linalg.solve(A, b)
                    if 0 <= t <= 1 and 0 <= u <= 1:  ##线段相交
                        z_pred = 1
                        # break
                # 不相交，计算最短距离
                if z_pred == 0:
                    dists = min([
                        point_to_segment_distance(p1, p3, p4),
                        point_to_segment_distance(p2, p3, p4),
                        point_to_segment_distance(p3, p1, p2),
                        point_to_segment_distance(p4, p1, p2),
                    ])
                    if dists < 2*self.d:
                        z_pred = 1
                        # break

                if z_pred == 1:
                    ## 如果z=1，那么此时再判断两个轮廓相交区域的占比情况，剔除占比过小的接触对
                    c_obj, c_obs = con_pair[3]
                    contour_obj = object_pose[c_obj]
                    contour_obs = self.obstacles[c_obs]
                    poly_obj = Polygon(contour_obj)
                    poly_obs = Polygon(contour_obs)
                    ## 只计算重合的区域占可移动物体轮廓的比例
                    if poly_obj.is_valid and poly_obs.is_valid and poly_obj.intersects(poly_obs):
                        inter_area = poly_obj.intersection(poly_obs).area
                        area1 = poly_obj.area
                        area2 = poly_obs.area
                        ratio1 = inter_area / area1 if area1 > 0 else 0
                        # ratio2 = inter_area / area2 if area2 > 0 else 0
                        ## 比例过低，认为此时不发生接触状态改变
                        if ratio1 < 0.05:
                            z_pred = 0
                        else:
                            z_pred = 1
                            break
                    else:
                        z_pred = 0  ##不发生重合

            # ##如果z=0,判断下一时刻潜在接触对的相交情况
            # if z_pred == 0:
            #     for con_pair in poten_con_pairs_next:
            #         obj_line_2, obs_line_2 = con_pair[0], con_pair[1]
            #         p1, p2 = obj_line_2
            #         p3, p4 = obs_line_2
            #
            #         d1 = p2 - p1
            #         d2 = p4 - p3
            #         A = np.array([d1, -d2]).T
            #         b = p3 - p1
            #         det = np.linalg.det(A)
            #
            #         if abs(det) > 1e-10:
            #             t, u = np.linalg.solve(A, b)
            #             if 0 <= t <= 1 and 0 <= u <= 1:  ##线段相交
            #                 z_pred = 1
            #                 # break
            #         # 不相交，计算最短距离
            #         if z_pred == 0:
            #             dists = min([
            #                 point_to_segment_distance(p1, p3, p4),
            #                 point_to_segment_distance(p2, p3, p4),
            #                 point_to_segment_distance(p3, p1, p2),
            #                 point_to_segment_distance(p4, p1, p2),
            #             ])
            #             if dists < 0.5 * self.d:
            #                 z_pred = 1
            #                 # break
            #
            #         if z_pred == 1:
            #             ## 如果z=1，那么此时再判断两个轮廓相交区域的占比情况，剔除占比过小的接触对
            #             c_obj, c_obs = con_pair[3]
            #             contour_obj = object_pose[c_obj]
            #             contour_obs = self.obstacles[c_obs]
            #             poly_obj = Polygon(contour_obj)
            #             poly_obs = Polygon(contour_obs)
            #             ## 只计算重合的区域占可移动物体轮廓的比例
            #             if poly_obj.is_valid and poly_obs.is_valid and poly_obj.intersects(poly_obs):
            #                 inter_area = poly_obj.intersection(poly_obs).area
            #                 area1 = poly_obj.area
            #                 area2 = poly_obs.area
            #                 ratio1 = inter_area / area1 if area1 > 0 else 0
            #                 # ratio2 = inter_area / area2 if area2 > 0 else 0
            #                 ## 比例过低，认为此时不发生接触状态改变
            #                 if ratio1 < 0.1:
            #                     z_pred = 0
            #                 else:
            #                     z_pred = 1
            #                     break
            #             else:
            #                 z_pred = 0  ##不发生重合

            return z_pred, obj_line_2, obs_line_2

    def gain_information(self, modes, Object):
        def ray_rectangle_intersection(p, di, ci):
            """
            判断点 p 沿方向 d 运动是否会进入矩形 c
            p: np.array([x,y]) 起点
            d: np.array([dx,dy]) 方向
            c: np.array([[xmin,ymin],[xmax,ymax]]) 矩形范围

            返回: (flag, distance)
                flag=True  → 会进入，distance=进入矩形的距离
                flag=False → 不会进入，distance=None
            """

            tmin, tmax = -np.inf, np.inf

            for i in range(2):  # 分别处理x和y
                if abs(di[i]) < 1e-10:  # 平行于该轴
                    if not (ci[0][i] <= p[i] <= ci[1][i]):
                        return False, None  # 永远不会进入
                else:
                    t1 = (ci[0][i] - p[i]) / di[i]
                    t2 = (ci[1][i] - p[i]) / di[i]
                    t_enter, t_exit = min(t1, t2), max(t1, t2)
                    tmin = max(tmin, t_enter)
                    tmax = min(tmax, t_exit)

            if tmax < tmin or tmax < 0:
                return False, None  # 不相交

            if tmin >= 0:  # 射线进入矩形
                return True, tmin * np.linalg.norm(di)
            else:  # 起点在矩形内部
                return True, 0.0

        if len(modes) == 0: ##无峰值，随机选择运动方向
            v_pred_2 = self.actions[np.random.randint(len(self.actions))]
        elif len(modes) == 1: ##单个峰值，可以寻找最优的路径
            v_pred_2 = np.array([0, 0])
            print(f'Find the uni-model and optimical trajectory!')
        else:
            self.Gain_information = []
            ##为了加快推理速度，只选择前1000个modes。
            if len(modes) > 1000:
                modes = sorted(modes, key=lambda x: x[1], reverse=True)[:1000]
            for a in self.actions:
                object = Object
                Distance_all = []
                Z_pred_all = []
                for c in modes:
                    pos = c[0]  ##峰值的位置，估计当参考点在此处时机器人如何运动
                    _, object_pose = self.add_movable_objects(object, pos)
                    _, _, poten_con_pairs_4 = self.select_pairs(a, object)
                    Dis = [] ##可能遇到的接触对应的ref_point运动范围
                    for pair in poten_con_pairs_4:
                        contact_range = pair[2]
                        IF_cutin, dis = ray_rectangle_intersection(pos, a, contact_range)
                        if IF_cutin == True:
                            Dis.append(dis)
                    if len(Dis) > 0:
                        Z_pred_all.append(1) ##此峰值位置和此运动方向下，会遇到障碍使传感器信号改变
                        Distance_all.append(min(Dis))
                    else:
                        Z_pred_all.append(0) ##此峰值位置和此运动方向下，不会遇到障碍
                        contact_range = np.array([[self.x_min, self.y_min],
                                              [self.x_max, self.y_max]])
                        d = ray_rectangle_intersection(pos, a, contact_range)
                        Distance_all.append(d)

                ##在运动方向a下，所有模式c的无障碍运动距离和障碍类型都已确定，计算此运动方向的信息增益
                #遇到不同的障碍类型的熵
                counts = Counter(Z_pred_all)  # 统计每个元素的频次
                total = len(Z_pred_all)  # 总数
                probs = np.array([count / total for count in counts.values()])  # 概率分布
                entropy = -np.sum(probs * np.log2(probs))  # 熵公式

                self.Gain_information.append(entropy) ##信息增益的衡量，一个是熵，还有一个与距离有关（未定）

            self.Gain_information = np.array(self.Gain_information).reshape(4, -1)
            v_pred_2 = self.actions[np.argmax(self.Gain_information)]   # 找到 a 中最大值的索引

        return v_pred_2

    def extract_circle_clusters(self, particles_5, radius=1.0):
        """
        1. 寻找最少覆盖圆
        2. 提取每个圆内的所有粒子点
        3. 计算每个簇的均值
        """
        remaining_indices = set(range(len(particles_5)))
        chosen_centers = []
        cluster_means = []

        tree = KDTree(particles_5)
        pts_array = particles_5

        # --- 第一步：贪心寻找覆盖圆 ---
        # 使用之前优化过的逻辑获取圆心
        while remaining_indices:
            best_center = None
            best_covered_indices = set()

            # 寻找当前最佳种子
            for i in remaining_indices:
                indices = tree.query_ball_point(pts_array[i], radius)
                current_covered = set(indices).intersection(remaining_indices)
                if len(current_covered) > len(best_covered_indices):
                    best_covered_indices = current_covered
                    best_center = pts_array[i]

            if best_center is None: break

            # 质心微调（可选，使圆位置更科学）
            for _ in range(2):
                current_pts = pts_array[list(best_covered_indices)]
                best_center = np.mean(current_pts, axis=0)
                new_indices = tree.query_ball_point(best_center, radius)
                best_covered_indices = set(new_indices).intersection(remaining_indices)

            # --- 第二步：提取该圆内的粒子并求均值 ---
            # 注意：这里提取的是此时此刻被该圆覆盖的粒子点
            actual_cluster_pts = pts_array[list(best_covered_indices)]
            if len(actual_cluster_pts) > 0:
                mean_pos = np.mean(actual_cluster_pts, axis=0)
                cluster_means.append(mean_pos)
                chosen_centers.append(best_center)

            # 移除已覆盖的点，继续下一轮
            remaining_indices -= best_covered_indices

        return np.array(chosen_centers), np.array(cluster_means)

    def gain_information_optimize(self, Object, clusters, particles, weights, hole_pose):
        def ray_rectangle_intersection(p, di, ci):
            """
            判断点 p 沿方向 d 运动是否会进入矩形 c
            p: np.array([x,y]) 起点
            d: np.array([dx,dy]) 方向
            c: np.array([[xmin,ymin],[xmax,ymax]]) 矩形范围

            返回: (flag, distance)
                flag=True  → 会进入，distance=进入矩形的距离
                flag=False → 不会进入，distance=None
            """

            tmin, tmax = -np.inf, np.inf

            for i in range(2):  # 分别处理x和y
                if abs(di[i]) < 1e-10:  # 平行于该轴
                    if not (ci[0][i] <= p[i] <= ci[1][i]):
                        return False, None  # 永远不会进入
                else:
                    t1 = (ci[0][i] - p[i]) / di[i]
                    t2 = (ci[1][i] - p[i]) / di[i]
                    t_enter, t_exit = min(t1, t2), max(t1, t2)
                    tmin = max(tmin, t_enter)
                    tmax = min(tmax, t_exit)

            if tmax < tmin or tmax < 0:
                return False, None  # 不相交

            if tmin >= 0:  # 射线进入矩形
                return True, tmin * np.linalg.norm(di)
            else:  # 起点在矩形内部
                return True, 0.0

        self.Gain_information_optimize = []
        object = Object
        Weight_left = []
        Particle_left = []
        Action_list_left, Path_list_left = [], []
        cluster_chosen = None
        clusters_array = []
        for p in clusters:
            Z_pred_all = []

            ## 针对当前的cluster规划到target_pose的最近轨迹
            action_list, path_list = self.plan_trajectory(p, hole_pose, Object)
            if len(action_list) == 0:
                # print(f"No valid path found for cluster at {p}. Skipping this cluster.")
                continue
            else:
                print(f"Found a valid path for cluster at {p}, action_list: {action_list}.")
            ## 评估这次运动带来的粒子的更新
            particles_moved = particles.copy()
            weights_moved = weights.copy()
            clusters_array.append(p)
            Particle_left.append(particles_moved)
            Weight_left.append(weights_moved)
            Action_list_left.append(np.array(action_list).reshape(-1, 2))
            Path_list_left.append(np.array(path_list).reshape(-1, 1))
            particles_moved_1 = np.zeros(0)
            weights_moved_1 = np.zeros(0)
            for action, dist in zip(action_list, path_list): ## 每一次运动带来的更新
                _, _, poten_con_pairs_4 = self.select_pairs(action, object)
                
                modes = self.find_local_maximum(particles_moved, weights_moved, radius=5)
                for c in modes:
                    pos = c[0]  ##峰值的位置，估计当参考点在此处时机器人如何运动
                    weight = c[1]
                    Dis = []  ##可能遇到的接触对应的ref_point运动距离
                    for pair in poten_con_pairs_4:
                        xmin, ymin = pair[2][0]
                        xmax, ymax = pair[2][1]
                        if abs(xmin - xmax) < 1.0: xmin -= 5.0; xmax += 5.0
                        if abs(ymin - ymax) < 1.0: ymin -= 5.0; ymax += 5.0  # 防止轮廓退化成一条直线
                        contact_range = np.array([[xmin, ymin],
                                                    [xmax, ymax]])
                        IF_cutin, dis = ray_rectangle_intersection(pos, action, contact_range)
                        if IF_cutin == True:  ##说明一定会遇到接触事件
                            if dis < dist: ##如果在没有运动结束就遇到接触事件
                                Z_pred_all.append(1) ##此峰值位置和此运动方向下，会遇到障碍使传感器信号改变
                                Dis.append(dis)
                                break
                    if len(Dis) == 0:
                        if np.max(action) == 1.0:
                            contact_range = np.array([[self.x_max+1e-10, 0.0], [0.0, self.y_max+1e-10]])
                            a1 = action.reshape(1, 2)
                            contour = np.dot(a1, contact_range)
                        else:
                            contact_range = np.array([[self.x_min+1e-10, 0.0], [0.0, self.y_min+1e-10]])
                            a1 = -action.reshape(1, 2)
                            contour = np.dot(a1, contact_range)
                        ## 此时对于距离的计算是到轮廓边缘的距离
                        dis = float(np.dot(action.reshape(1,2), (contour - pos.reshape(1, 2)).T))
                        if dis < dist: ##如果在没有运动结束就超出范围
                            Z_pred_all.append(1) ##此峰值位置和此运动方向下，不会遇到障碍，而会超出轮廓范围(这里也认为是1)
                        else:
                            particles_moved_1 = np.append(particles_moved_1, pos).reshape(-1, 2)
                            weights_moved_1 = np.append(weights_moved_1, weight)
                    
                ## 更新粒子集合和权重，进行下一次运动的评估
                particles_moved = particles_moved_1
                weights_moved = weights_moved_1
                if particles_moved.shape[0] == 0:
                    print(f"All particles have been filtered out after action {action}. Breaking out of the loop.")
                    break
                displacement = dist * action 
                particles_moved += displacement ##在粒子更新后进行位置的更新
                particles_moved_1 = np.zeros(0)
                weights_moved_1 = np.zeros(0)

            ## 计算粒子的加权数目
            N_particles_all = np.ones(particles.shape[0])
            ##在运动方向a下，所有模式c的无障碍运动距离和障碍类型都已确定，计算此运动方向的信息增益
            # 遇到不同的障碍类型的熵
            Z_pred_all = np.array(Z_pred_all)
            counts = [np.sum(Z_pred_all) + 1e-10,
                          np.sum(N_particles_all)-np.sum(Z_pred_all) + 1e-10]
            total = np.sum(N_particles_all)  # 总数
            probs = np.array([count / total for count in counts])  # 概率分布
            entropy = -np.sum(probs * np.log2(probs))  # 熵公式

            self.Gain_information_optimize.append(entropy)

        if len(self.Gain_information_optimize) == 0:
            print('The optimization moving trend is unsuitable!')
            self.Gain_information_optimize = np.array([0]).reshape(-1, 1)
        else:
            self.Gain_information_optimize = np.array(self.Gain_information_optimize).reshape(-1, 1)
            ## 选择最大的增益对应的cluster
            clusters_array = np.array(clusters_array).reshape(-1, 2)
            cluster_chosen = clusters_array[np.argmax(self.Gain_information_optimize)]  # 找到 a 中最大值的索引
            ## 打印熵的情况，判断动作方向的选择是否合理
            print(f'The Gain_information is: {self.Gain_information_optimize}, The chosen cluster is: {cluster_chosen}')
            print(f'The chosen cluster is: {cluster_chosen}, the Gain_information_optimize is: {self.Gain_information_optimize[np.argmax(self.Gain_information_optimize)]}')
            Action_list_left = Action_list_left[np.flatnonzero((clusters_array == cluster_chosen).all(axis=1))[0]]  # 找到a对应的粒子集合
            Path_list_left = Path_list_left[np.flatnonzero((clusters_array == cluster_chosen).all(axis=1))[0]]

        return cluster_chosen, Action_list_left, Path_list_left

    ##模拟物体的运动，并更新权重-------------------------------------
    def weights_updates(self, action, objects, z_obs_list, particles_4, weights_4):
        T = len(z_obs_list)
        for i in range(particles_4.shape[0]):
            meet_obstacle = False ##由于物体首次接收到状态改变就进入更新，因此观测序列之前都是未碰到障碍物的
            for t in range(T):
                obs = z_obs_list[t]
                if meet_obstacle == False:
                    offset = particles_4[i] - action * self.d *(T-t)
                    z_pred, _, _ = self.obs_pred(action=action,
                                                 Object=objects, ref_point_3=offset)

                    # 更新相应particle点的权重
                    weights_4[i] = weights_4[i] * self.likelihood(obs, z_pred)
                    if z_pred == 1 or z_pred == -1: ##遇到障碍物或者超出范围
                        meet_obstacle = True
                else: ##碰到障碍物，物体就不动了
                    weights_4[i] = weights_4[i] * (self.likelihood(obs=0, pred=-1)**(T-t))
                    t=T

        weights_4 /= np.sum(weights_4)

        return weights_4

    def weights_updates_dis(self, action, Movable_objects_1, z_obs_list, Pos_list_1, particles_5, weights_5):
        T = len(z_obs_list)
        objects = Movable_objects_1
        for i in range(particles_5.shape[0]): ##对每一个粒子进行判断与更新
            meet_obstacle = False ##由于物体首次接收到状态改变就进入更新，因此观测序列之前都是未碰到障碍物的
            print(f'i={i}/{particles_5.shape[0]}')
            for t in range(T):
                obs = z_obs_list[t]
                delta = Pos_list_1[-1]-Pos_list_1[t]
                if meet_obstacle == False:
                    offset = particles_5[i] - delta
                    z_pred, _, _ = self.obs_pred(action=action,
                                                 Object=objects, ref_point_3=offset)

                    # 更新相应particle点的权重
                    weights_5[i] = weights_5[i] * self.likelihood(obs, z_pred)
                    if z_pred == 1 or z_pred == -1: ##遇到障碍物或者超出范围
                        meet_obstacle = True
                else: ##碰到障碍物，物体就不动了
                    weights_5[i] = weights_5[i] * (self.likelihood(obs=0, pred=-1)**(T-t))
                    t=T

        weights_5 /= np.sum(weights_5)

        return weights_5

    ##机械臂执行的主程序
    def main_line_perception(self, Z_obs_2, v_pred_3, Movable_objects_3):
        particle_range_3 = []
        z_obs_2 = Z_obs_2[-1]
        x_min, x_max, y_min, y_max = 0,0,0,0

        movable_objects_3 = Movable_objects_3
        if z_obs_2 == -1:
            if abs(v_pred_3[1]) == 1:
                d_y = 10.0
                x_min = self.x_min
                x_max = self.x_max
                if v_pred_3[1] == 1:
                    y_min = self.y_max - d_y
                    y_max = self.y_max + d_y
                else:
                    y_min = self.y_min - d_y
                    y_max = self.y_min + d_y
            if abs(v_pred_3[0]) == 1:
                d_x = 10.0
                y_min = self.y_min
                y_max = self.y_max
                if v_pred_3[0] == 1:
                    x_min = self.x_max - d_x
                    x_max = self.x_max + d_x
                else:
                    x_min = self.x_min - d_x
                    x_max = self.x_min + d_x
            con_range_3 = np.array([[x_min, y_min], [x_max, y_min],
                                  [x_max, y_max], [x_min, y_max]])
            particle_range_3.append(con_range_3)

        if z_obs_2 == 1:
            # 潜在的接触对
            _, _, poten_con_pairs = self.select_pairs(v_pred_3, movable_objects_3)
            # 更新粒子分布(粒子分布首先根据上一次筛选得到的范围平移后再与新的潜在接触范围做交集)
            for con_pair in poten_con_pairs:
                xmin, ymin = con_pair[2][0]
                xmax, ymax = con_pair[2][1]
                if abs(xmin - xmax) < 0.5: xmin -= 4.0; xmax += 4.0
                if abs(ymin - ymax) < 0.5: ymin -= 4.0; ymax += 4.0  # 防止轮廓退化成一条直线
                con_range_3 = np.array([[xmin, ymin], [xmax, ymin],
                                      [xmax, ymax], [xmin, ymax]])
                particle_range_3.append(con_range_3)
        print(f'particle_range={particle_range_3}')

        return particle_range_3


def scene_plot(scene, xs, ys, object_mask, obstacles, object_pos, ref_point, particles):
    # === 可视化 ===
    # ---------- 可视化 ----------
    fig, ax = plt.subplots(figsize=(7, 6))

    # 背景：静态场景（白=空闲，黑=静态障碍）
    im = ax.imshow(
        scene, extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower", cmap="gray_r", interpolation="nearest"
    )

    # 叠加：可移动物体的红色半透明网格
    # 构造 RGBA 覆盖层：红色(1,0,0,alpha)；非掩码处 alpha=0
    overlay = np.zeros((object_mask.shape[0], object_mask.shape[1], 4), dtype=float)
    overlay[..., 0] = 1.0  # R
    overlay[..., 3] = object_mask * 0.55  # Alpha
    ax.imshow(
        overlay, extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower", interpolation="nearest"
    )

    # 粒子分布
    plt.scatter(particles[:, 0], particles[:, 1], c=weights,  # 用归一化权重控制颜色
                 cmap='coolwarm', s=10, label="Particles")

    # 轮廓线（辅助查看形状）
    for sp in obstacles:
        ax.plot(np.r_[sp[:, 0], sp[0, 0]], np.r_[sp[:, 1], sp[0, 1]], 'k-', lw=1.2, label='_nolegend_')
    for mp in object_pos:
        ax.plot(np.r_[mp[:, 0], mp[0, 0]], np.r_[mp[:, 1], mp[0, 1]], 'r-', lw=1.2, label='_nolegend_')

    # 参考点
    ax.scatter([ref_point[0]], [ref_point[1]], c='g', s=50, marker='*')

    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Add movable obstacles (red cells) and reference point (green *)')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(loc='upper right')
    plt.show()


# ===== 示例 =====
if __name__ == "__main__":
    # # 障碍物1：矩形
    # rect = np.array([[20, 20], [60, 20], [60, 60], [20, 60]])
    # # 障碍物2：三角形
    # tri = np.array([[80, 30], [120, 50], [100, 90]])
    # obstacles = [rect, tri]
    # import pickle
    # pkl_path = f'socket_contours.pkl'
    # with open(pkl_path, "rb") as f:
    #     loaded = pickle.load(f)
    # obstacles =loaded
    obstacles = [
            np.array([[0.0, 0.0], [0.0, -180.0], [20.0, -180.0], [20.0, -20.0], [160.0, -20.0],
                      [160.0, 0.0]])
        ]

    # 可移动物体由多个封闭轮廓组成（相对参考点坐标）
    Movable_objects = [
            np.array([[-4.5, 85.0], [-4.5, -85.0], [4.5, -85.0], [4.5, 85.0]], dtype=float),  # 小矩形1
        ]
    # Movable_objects = [np.array([[-0.75, 5.98], [0.75, 5.98], [0.75, 12.38], [-0.75, 12.38]])]
    # pkl_path = f'insertor_two.pkl'
    # with open(pkl_path, "rb") as f:
    #     loaded = pickle.load(f)
    # Movable_objects = loaded
    movable_objects = Movable_objects

    ref_point = np.array([50, -125], dtype=float) ##这里的参考点坐标要随即在场景中选取

    ## 构建仿真场景
    curve_dis = np.array([25.0, 25.0, 25.0, 25.0]) ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
    M_planning = Motion_planning(d=2.0, padding=curve_dis)
    scene, xs, ys = M_planning.discretize_scene(obstacles)
    # 把可移动物体的坐标表示到场景中
    object_mask, object_pos = M_planning.add_movable_objects(movable_objects, ref_point)
    # 初始化粒子分布
    particle_range = np.array([[xs[0], ys[0]], [xs[-1], ys[0]],
                               [xs[-1], ys[-1]], [xs[0], ys[-1]]])
    particles, weights = M_planning.generate_particles_in_polygon([particle_range], N=2000)
    #检测峰值，选择相应的方向
    peaks = M_planning.find_local_maximum(particles, weights, radius=5)

    num = 0
    while not len(peaks)==1: ##如果峰值数量不是1,那么进入主动探索阶段
        ##绘图
        scene_plot(scene, xs, ys, object_mask, obstacles, object_pos, ref_point, particles)

        v_pred = M_planning.gain_information(peaks, Movable_objects)

        movable_objects = Movable_objects

        ##沿着选定的方向运动，直至z=1
        z = 0
        Z_obs = []
        Pos_list = np.zeros(0)


        while z == 0:
            # 检测当前的接触状态（目前使用预测的接触状态）
            Pos_list = np.append(Pos_list, ref_point).reshape(-1, 2)
            z_obs, obj_line, obs_line = M_planning.obs_pred(action=v_pred,
                                        Object=movable_objects, ref_point_3=ref_point)
            Z_obs.append(z_obs)  # 收集传感器状态序列
            ref_point += v_pred * M_planning.d  # 每步的步长为d

            if z_obs == -1:
                # 更新一下物体的位置
                object_mask, object_pos = M_planning.add_movable_objects(Movable_objects, ref_point)
                particle_range = []
                xmin, xmax, ymin, ymax = 0,0,0,0
                if abs(v_pred[1]) == 1:
                    delta_x = 0.0
                    delta_y = 10.0
                    xmin = M_planning.x_min
                    xmax = M_planning.x_max
                    if v_pred[1] == 1:
                        ymin = M_planning.y_max - delta_y
                        ymax = M_planning.y_max + delta_y
                    else:
                        ymin = M_planning.y_min - delta_y
                        ymax = M_planning.y_min + delta_y
                if abs(v_pred[0]) == 1:
                    delta_x = 10.0
                    delta_y = 0.0
                    ymin = M_planning.y_min
                    ymax = M_planning.y_max
                    if v_pred[0] == 1:
                        xmin = M_planning.x_max - delta_x
                        xmax = M_planning.x_max + delta_x
                    else:
                        xmin = M_planning.x_min - delta_x
                        xmax = M_planning.x_min + delta_x

                con_range = np.array([[xmin, ymin], [xmax, ymin],
                                      [xmax, ymax], [xmin, ymax]])
                particle_range.append(con_range)
                if num == 0:
                    particles, weights = M_planning.generate_particles_in_polygon(particle_range, N=2000)
                else:
                    particles, weights = M_planning.update_particles(particles, weights, particle_range,
                                                                     shift_3=Pos_list[-1]-Pos_list[0])
                ##更新图像
                scene_plot(scene, xs, ys, object_mask, obstacles, object_pos, ref_point, particles)
                z = 1

            if z_obs == 1:
                # 更新一下物体的位置
                object_mask, object_pos = M_planning.add_movable_objects(Movable_objects, ref_point)
                # 潜在的接触对
                _, _, poten_con_pairs = M_planning.select_pairs(v_pred, movable_objects)
                # 更新粒子分布(粒子分布首先根据上一次筛选得到的范围平移后再与新的潜在接触范围做交集)
                particle_range = []
                for con_pair in poten_con_pairs:
                    xmin, ymin = con_pair[2][0]
                    xmax, ymax = con_pair[2][1]
                    if abs(xmin - xmax)<0.5: xmin -= 4.0; xmax += 4.0
                    if abs(ymin - ymax)<0.5: ymin -= 4.0; ymax += 4.0  # 防止轮廓退化成一条直线
                    con_range = np.array([[xmin, ymin], [xmax, ymin],
                                          [xmax, ymax], [xmin, ymax]])
                    particle_range.append(con_range)
                if num == 0:
                    particles, weights = M_planning.generate_particles_in_polygon(particle_range, N=2000)
                else:
                    particles, weights = M_planning.update_particles(particles, weights, particle_range,
                                                                     shift_3=Pos_list[-1] - Pos_list[0])


                ##更新图像
                scene_plot(scene, xs, ys, object_mask, obstacles, object_pos, ref_point, particles)
                z = 1


        # 更新权重
        # weights = M_planning.weights_updates(action=v_pred,
        #                                      objects=movable_objects, z_obs_list=Z_obs,
        #                                      particles=particles, weights=weights)
        weights = M_planning.weights_updates_dis(action=v_pred, Movable_objects_1=Movable_objects, z_obs_list=Z_obs,
                                                 Pos_list_1=Pos_list, particles_5=particles, weights_5=weights)

        ## 针对运动到场景边缘外的情况，回退到上一次运动前
        if Z_obs[-1]==-1:
            shift = -v_pred * M_planning.d * len(Z_obs)
            ref_point += shift
            particles = particles + shift
            # 更新一下物体的位置
            object_mask, object_pos = M_planning.add_movable_objects(Movable_objects, ref_point)
        # 检测峰值，选择相应的方向
        peaks = M_planning.find_local_maximum(particles, weights, radius=5)
        print('len(peaks)=', len(peaks))
        num += 1
