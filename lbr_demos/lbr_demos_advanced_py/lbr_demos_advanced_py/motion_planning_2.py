import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import splrep, splev
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.transform import Rotation as R, Slerp

"""
完善主动探索、信息增益后的代码
251206:已完成信息增益的改进，需进一步完成主动探索策略的改进，当粒子数目少时执行目标轨迹探索
"""

class Motion_planning():
    def __init__(self, d, padding):
        self.Weight_free = None
        self.Pos_free = None
        self.normals_obstacles = None
        self.normals_objects = None
        self.Gain_information = None
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
        # all_equal = np.all(weights_2 == weights_2[0])
        # if all_equal and len(particles_2) > 10:
        #     return peaks_1
        # else:
        #     # 粒子的权重大于weight_thre的都这值为局部最优
        #     w_thresh = 1.0 / (len(particles_2) * 2)
        #     mask = weights_2 >= w_thresh
        #     particles_maximum = particles_2[mask]
        #     weights_maximum = weights_2[mask]
        #     for i in range(len(particles_maximum)):
        #         ix, iy = particles_maximum[i]
        #         val = weights_maximum[i]
        #         peaks_1.append([np.array([ix, iy]), val])
        #
        #     return peaks_1


    def update_particles(self, particles_3, weights_3, polygons, z_obs_list, shift_3=np.array([0, 0])):
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
        weights_3 = weights_3[mask]

        if len(particles_3) == 0:
            print("No particles left")
            return np.zeros((0, 2)), np.zeros(0)


        # 2. 平移粒子
        particles_shifted = particles_3 + np.array(shift_3, dtype=np.float64)

        # 3. 判断是否在任意多边形内
        inside_mask_total = np.zeros(len(particles_shifted), dtype=bool)
        for poly in polygons:
            path = Path(poly)
            inside_mask_total |= path.contains_points(particles_shifted)

        # 4. 保留在多边形内的粒子
        new_particles = particles_shifted[inside_mask_total]
        new_weights = weights_3[inside_mask_total]
        print(f'len(new_particles) before filtering= {len(new_particles)}')

        # ## 根据先前信息增益计算的结果缩减粒子的数目，得到更少的需要迭代的粒子(251215添加)(不完善，需进一步完善)
        # if z_obs_list[-1] == 1:
        #     particles_remain = np.array(self.Pos_free[0]).reshape(-1, 2)
        #     weights_remain = np.array(self.Weight_free[0]).reshape(-1, 2)
        # else:
        #     particles_remain = np.array(self.Pos_free[1]).reshape(-1, 2)
        #     weights_remain = np.array(self.Weight_free[1]).reshape(-1, 2)
        # particles_shifted = particles_remain + np.array(shift_3)
        # inside_mask_total = np.zeros(len(particles_shifted), dtype=bool)
        # for poly in polygons:
        #     path = Path(poly)
        #     inside_mask_total |= path.contains_points(particles_shifted)
        # new_particles_1 = particles_shifted[inside_mask_total]
        # new_weights_1 = weights_remain[inside_mask_total]
        # if len(new_particles_1) > 10:
        #     new_particles = new_particles_1
        #     new_weights = new_weights_1
        # else:
        #     print("No particles left after filtering")
        # print(f'len(new_particles) after filtering= {len(new_particles)}')

        # N = new_particles.shape[0]
        # new_weights = np.ones(N) / N

        return new_particles, new_weights, particles_3

    def resample_particles(self, new_particles, new_weights, particles_3):
        # 5. 计算剩余的粒子的范围
        std = np.std(new_particles, axis=0)
        dis = np.linalg.norm(std)
        print(f'std(new_particles) before recutting={std})')
        # 7. 重采样
        # # 对比组，重采样的数据量固定
        # if dis > 3.0 and len(new_particles) < 50:
        # # 随机角度
        #     theta = np.random.uniform(0, 2 * np.pi, size=4)
        #     # 随机半径 (平方根保证均匀分布在圆内)
        #     r=1
        #     radius = r * np.sqrt(np.random.uniform(0, 1, size=4))
        #     # 极坐标转直角坐标
        #     Points = np.zeros(0)
        #     for p in new_particles:
        #         x = p[0] + radius * np.cos(theta)
        #         y = p[1] + radius * np.sin(theta)
        #         points = np.column_stack((x, y))
        #         Points = np.append(Points, points)
        #         Points.reshape(-1, 2)
        #     new_particles = np.append(new_particles, Points).reshape(-1, 2)
        #     new_particles = np.unique(new_particles, axis=0)

        # 每次粒子更新完成之后，都要进行重采样，剩余的粒子数越多，重采样的点应该越少
        if len(new_particles) > 300:
            scale = int((len(particles_3) // len(new_particles)) / 3)
        elif 100 < len(new_particles) < 300:
            scale = int((len(particles_3) // len(new_particles)) / 2)
        elif 50 < len(new_particles) < 100:
            scale = len(particles_3) // len(new_particles)
        else:
            scale = 2 * (len(particles_3) // len(new_particles))
        if dis > 2.5:
            # 随机角度
            theta = np.random.uniform(0, 2 * np.pi, size=scale)
            # 随机半径 (平方根保证均匀分布在圆内)
            r=1.0
            radius = r * np.sqrt(np.random.uniform(0, 1, size=scale))
            # 极坐标转直角坐标
            Points = np.zeros(0)
            for p in new_particles:
                x = p[0] + radius * np.cos(theta)
                y = p[1] + radius * np.sin(theta)
                points = np.column_stack((x, y))
                Points = np.append(Points, points)
                Points = Points.reshape(-1, 2)
            new_particles = np.append(new_particles, Points).reshape(-1, 2)
            N = Points.shape[0]
            weights_point = np.ones(N) / N
            new_weights = np.append(new_weights, weights_point)
            # new_particles = np.unique(new_particles, axis=0)

        new_weights /= np.sum(new_weights)

        print(f'len(new_particles) after recutting= {len(new_particles)}')

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
        self.normals_obstacles = self.n_vector(self.obstacles, type_1='stay')
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
            _, _, poten_con_pairs_2 = self.select_pairs(action, object_pose) ## 这里需要判断接触对的重合关系，因此可移动物体的位置采用当前的坐标，而不是相对参考点的坐标
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
                    if dists < 0.5:
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
                        if ratio1 < 0.1:
                            z_pred = 0
                        elif 0.1 < ratio1 < 0.8:
                            z_pred = (ratio1 - 0.1)/0.7
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

    ## 251204修改，与原版相比主要改进信息增益计算考虑例子的权重
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

        if len(modes) == 2000: ##无峰值，随机选择运动方向(251205实验)
            v_pred_2 = self.actions[np.random.randint(len(self.actions))]
            print(f'v_pred={v_pred_2}')
            print("Is the selected velocity suitable? Press 'q' and Enter to continue...")
            while True:
                user_in = input()
                if user_in == 'q':
                    break
                elif user_in == 'w':
                    try:
                        new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                        new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                        v_pred_2 = np.array([new_val_0, new_val_1])
                        print(f"self.v_pred 已更新为 {v_pred_2}")
                    except ValueError:
                        print("输入无效，请输入数值。")
                else:
                    print("无效输入，请输入 'q' 或 'w'")

            self.Gain_information = []
            self.Pos_free = []
            object = Object
            Z_pred_all = []
            Weight_all = []
            Distance_all = []
            Pos = []
            for c in modes:
                pos = c[0]  ##峰值的位置，估计当参考点在此处时机器人如何运动
                weight = c[1]
                Weight_all.append(weight)
                Pos.append(pos)
                _, _, poten_con_pairs_4 = self.select_pairs(v_pred_2, object)
                Dis = []  ##可能遇到的接触对应的ref_point运动范围
                for pair in poten_con_pairs_4:
                    xmin, ymin = pair[2][0]
                    xmax, ymax = pair[2][1]
                    if abs(xmin - xmax) < 1.0: xmin -= 2.5; xmax += 2.5
                    if abs(ymin - ymax) < 1.0: ymin -= 2.5; ymax += 2.5  # 防止轮廓退化成一条直线
                    contact_range = np.array([[xmin, ymin],
                                              [xmax, ymax]])
                    IF_cutin, dis = ray_rectangle_intersection(pos, v_pred_2, contact_range)
                    if IF_cutin == True:  ##说明一定会遇到接触事件
                        Dis.append(dis)
                if len(Dis) > 0:
                    Z_pred_all.append(1)  ##此峰值位置和此运动方向下，会遇到障碍使传感器信号改变
                    Distance_all.append(min(Dis))
                else:
                    Z_pred_all.append(-1)  ##此峰值位置和此运动方向下，不会遇到障碍，而会超出轮廓范围
                    if np.max(v_pred_2) == 1.0:
                        contact_range = np.array([[self.x_max, 0.0], [0.0, self.y_max]])
                        a1 = v_pred_2.reshape(1, 2)
                        contour = np.dot(a1, contact_range)
                    else:
                        contact_range = np.array([[self.x_min, 0.0], [0.0, self.y_min]])
                        a1 = -v_pred_2.reshape(1, 2)
                        contour = np.dot(a1, contact_range.T)
                    ## 此时对于距离的计算是到轮廓边缘的距离
                    d = float(np.dot(a1, (contour - pos.reshape(1, 2)).T))
                    Distance_all.append(d)
            ## 计算粒子的加权数目
            N_particles_all = np.ones(2000)
            ##在运动方向a下，所有模式c的无障碍运动距离和障碍类型都已确定，计算此运动方向的信息增益
            # 遇到不同的障碍类型的熵
            Z_pred_all = np.array(Z_pred_all)
            counts = [np.sum(N_particles_all[Z_pred_all == 1])+1e-10, np.sum(N_particles_all[Z_pred_all == -1])+1e-10]
            total = np.sum(N_particles_all)  # 总数
            probs = np.array([count / total for count in counts])  # 概率分布
            entropy = -np.sum(probs * np.log2(probs))  # 熵公式

            Pos = np.array(Pos).reshape(-1, 2)
            Weight_all = np.array(Weight_all)
            self.Pos_free = [Pos[Z_pred_all == 1], Pos[Z_pred_all == -1]]  # 对于在此运动方向下不同接触事件对应的粒子位置的集合
            self.Weight_free = [Weight_all[Z_pred_all == 1], Weight_all[Z_pred_all == -1]]
            self.Gain_information.append(entropy)  ##信息增益的衡量，一个是熵，还有一个与距离有关（未定）

        elif len(modes) == 1: ##峰值数量不多时，判断粒子的分布是否集中在有限数量内
            v_pred_2 = np.array([0, 0])
            print(f'Find the uni-model and optimal trajectory!')
        else:
            self.Gain_information = []
            self.Pos_free = []
            self.Weight_free = []
            self.Dis_var = []
            for a in self.actions:
                object = Object
                Z_pred_all = []
                Weight_all = []
                Distance_all = []
                Pos = []
                _, _, poten_con_pairs_4 = self.select_pairs(a, object)
                for c in modes:
                    pos = c[0]  ##峰值的位置，估计当参考点在此处时机器人如何运动
                    weight = c[1]
                    Weight_all.append(weight)
                    Pos.append(pos)
                    Dis = [] ##可能遇到的接触对应的ref_point运动距离
                    for pair in poten_con_pairs_4:
                        xmin, ymin = pair[2][0]
                        xmax, ymax = pair[2][1]
                        if abs(xmin - xmax) < 1.0: xmin -= 2.5; xmax += 2.5
                        if abs(ymin - ymax) < 1.0: ymin -= 2.5; ymax += 2.5  # 防止轮廓退化成一条直线
                        contact_range = np.array([[xmin, ymin],
                                                [xmax, ymax]])
                        IF_cutin, dis = ray_rectangle_intersection(pos, a, contact_range)
                        if IF_cutin == True: ##说明一定会遇到接触事件
                            Dis.append(dis)
                    if len(Dis) > 0:
                        Z_pred_all.append(1) ##此峰值位置和此运动方向下，会遇到障碍使传感器信号改变
                        Distance_all.append(min(Dis))
                    else:
                        Z_pred_all.append(-1) ##此峰值位置和此运动方向下，不会遇到障碍，而会超出轮廓范围
                        if np.max(a) == 1.0:
                            contact_range = np.array([[self.x_max+1e-10, 0.0], [0.0, self.y_max+1e-10]])
                            a1 = a.reshape(1, 2)
                            contour = np.dot(a1, contact_range)
                        else:
                            contact_range = np.array([[self.x_min+1e-10, 0.0], [0.0, self.y_min+1e-10]])
                            a1 = -a.reshape(1, 2)
                            contour = np.dot(a1, contact_range.T)
                        ## 此时对于距离的计算是到轮廓边缘的距离
                        d = float(np.dot(a1, (contour - pos.reshape(1, 2)).T))
                        Distance_all.append(d)
                ## 计算粒子的加权数目
                if max(Weight_all) - min(Weight_all) < 1e-10:
                    N_particles_all = np.ones(len(modes))
                else:
                    N_particles_all = (Weight_all - min(Weight_all)) / (max(Weight_all) - min(Weight_all)) + 1
                    N_particles_all = np.array(N_particles_all)
                ##在运动方向a下，所有模式c的无障碍运动距离和障碍类型都已确定，计算此运动方向的信息增益
                #遇到不同的障碍类型的熵
                Z_pred_all = np.array(Z_pred_all)
                Distance_all = np.array(Distance_all)
                counts = [np.sum(N_particles_all[Z_pred_all == 1])+1e-10, np.sum(N_particles_all[Z_pred_all == -1])+1e-10]
                total = np.sum(N_particles_all)  # 总数
                probs = np.array([count / total for count in counts])  # 概率分布
                entropy = -np.sum(probs * np.log2(probs))  # 熵公式
                variance = np.std(Distance_all) # 距离的变化范围
                Pos = np.array(Pos).reshape(-1, 2)
                pos_free = [Pos[Z_pred_all == 1], Pos[Z_pred_all == -1]] # 对于在此运动方向下不同接触事件对应的粒子位置的集合
                Weight_all = np.array(Weight_all)
                weight_free = [Weight_all[Z_pred_all == 1], Weight_all[Z_pred_all == -1]]

                self.Gain_information.append(entropy) ##信息增益的衡量，一个是熵，还有一个与距离有关（未定）
                self.Pos_free.append(pos_free)
                self.Weight_free.append(weight_free)
                self.Dis_var.append(variance)

            self.Gain_information = np.array(self.Gain_information).reshape(4, -1)
            self.Dis_var = np.array(self.Dis_var).reshape(4, -1)
            if np.max(self.Gain_information) < 1e-2:
                print("The all entropy are low and we choose the distance flag.")
                print(f'The Gain_information is: {self.Gain_information}')
                v_pred_2 = self.actions[np.argmax(self.Dis_var)]  # 找到 a 中最大值的索引
                print(f'The Dis_var is: {self.Dis_var}, The current velocity is: {v_pred_2}')
            else:
                v_pred_2 = self.actions[np.argmax(self.Gain_information)]   # 找到 a 中最大值的索引
                ## 打印熵的情况，判断动作方向的选择是否合理
                print(f'The Gain_information is: {self.Gain_information}, The current velocity is: {v_pred_2}')
                print("Is the selected velocity suitable? Press 'q' and Enter to continue...")
            # v_pred_2 = self.actions[np.argmax(self.Gain_information)]  # 找到 a 中最大值的索引
            # ## 打印熵的情况，判断动作方向的选择是否合理
            # print(f'The Gain_information is: {self.Gain_information}, The current velocity is: {v_pred_2}')
            # print("Is the selected velocity suitable? Press 'q' and Enter to continue...")
            while True:
                user_in = input()
                if user_in == 'q':
                    break
                elif user_in == 'w':
                    try:
                        new_val_0 = float(input("请输入新的 v_pred[0] 值: "))
                        new_val_1 = float(input("请输入新的 v_pred[1] 值: "))
                        v_pred_2 = np.array([new_val_0, new_val_1])
                        print(f"self.v_pred 已更新为 {v_pred_2}")
                    except ValueError:
                        print("输入无效，请输入数值。")
                else:
                    print("无效输入，请输入 'q' 或 'w'")
            self.Pos_free = self.Pos_free[np.flatnonzero((self.actions == v_pred_2).all(axis=1))[0]] # 找到a对应的粒子集合
            self.Weight_free = self.Weight_free[np.flatnonzero((self.actions == v_pred_2).all(axis=1))[0]]

        return v_pred_2

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
        # 根据v_pred的方向，对应不同的Movable_objects_1的范围
        # if action[1] == 1:
        #     objects = [Movable_objects_1[1]]
        # elif action[1] == -1:
        #     objects = [Movable_objects_1[0]]
        # else:
        #     objects = Movable_objects_1
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
        #根据v_pred的方向，对应不同的movable_objects的范围
        # if v_pred_3[1] == 1:
        #     movable_objects_3 = [Movable_objects_3[1]]
        # elif v_pred_3[1] == -1:
        #     movable_objects_3 = [Movable_objects_3[0]]
        # else:
        #     movable_objects_3 = Movable_objects_3
        movable_objects_3 = Movable_objects_3
        if z_obs_2 == -1:
            if abs(v_pred_3[1]) == 1:
                d_y = 8.0
                x_min = self.x_min
                x_max = self.x_max
                if v_pred_3[1] == 1:
                    y_min = self.y_max - d_y
                    y_max = self.y_max + d_y
                else:
                    y_min = self.y_min - d_y
                    y_max = self.y_min + d_y
            if abs(v_pred_3[0]) == 1:
                d_x = 8.0
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
                if abs(xmin - xmax) < 1.0: xmin -= 2.5; xmax += 2.5
                if abs(ymin - ymax) < 1.0: ymin -= 2.5; ymax += 2.5  # 防止轮廓退化成一条直线
                con_range_3 = np.array([[xmin, ymin], [xmax, ymin],
                                      [xmax, ymax], [xmin, ymax]])
                particle_range_3.append(con_range_3)

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

    # # 轮廓线（辅助查看形状）
    # for sp in obstacles:
    #     ax.plot(np.r_[sp[:, 0], sp[0, 0]], np.r_[sp[:, 1], sp[0, 1]], 'k-', lw=1.2, label='_nolegend_')
    for mp in object_pos:
        ax.plot(np.r_[mp[:, 0], mp[0, 0]], np.r_[mp[:, 1], mp[0, 1]], 'r-', lw=1.2, label='_nolegend_')

    # 参考点
    ax.scatter([ref_point[0]], [ref_point[1]], c='g', s=50, marker='*')

    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    # ax.set_title('Add movable obstacles (red cells) and reference point (green *)')
    plt.xticks(fontsize=1)
    plt.yticks(fontsize=1)
    ax.legend(loc='upper right')
    plt.show()


# ===== 示例 =====
if __name__ == "__main__":
    # # 障碍物1：矩形
    # rect = np.array([[20, 20], [60, 20], [60, 60], [20, 60]])
    # # 障碍物2：三角形
    # tri = np.array([[80, 30], [120, 50], [100, 90]])
    # obstacles = [rect, tri]
    import pickle
    pkl_path = f'socket_contours.pkl'
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    obstacles =loaded
    R = np.array([[0, -1], [1, 0]])  # 顺时针90°旋转矩阵
    rotated = []
    for c in obstacles:
        rotated.append(c @ R.T)  # 矩阵乘法
    obstacles = rotated

    # 可移动物体由多个封闭轮廓组成（相对参考点坐标）
    Movable_objects = [
        np.array([[-7, -3.2], [-5.5, -3.2], [-5.5, 3.2], [-7, 3.2]], dtype=float),  # 小矩形1
        np.array([[5.5, -3.2], [7, -3.2], [7, 3.2], [5.5, 3.2]], dtype=float)  # 小矩形2（与1相对位置固定）
    ]
    # Movable_objects = [np.array([[-0.75, 5.98], [0.75, 5.98], [0.75, 12.38], [-0.75, 12.38]])]
    # pkl_path = f'insertor_two.pkl'
    # with open(pkl_path, "rb") as f:
    #     loaded = pickle.load(f)
    # Movable_objects = loaded
    R = np.array([[0, -1], [1, 0]])  # 顺时针90°旋转矩阵
    rotated = []
    for c in Movable_objects:
        rotated.append(c @ R.T)  # 矩阵乘法
    Movable_objects = rotated
    movable_objects = Movable_objects

    ref_point = np.array([-25, 55], dtype=float) ##这里的参考点坐标要随即在场景中选取

    ## 构建仿真场景
    curve_dis = np.array([10.4, 35.7, 15.5, 15.5]) ##[delta_x_min,delta_x_max, delta_y_min, delta_y_max]
    M_planning = Motion_planning(d=0.25, padding=curve_dis)
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

        # ##根据v_pred的方向，对应不同的movable_objects的范围
        # if v_pred[1] == 1:
        #     movable_objects = [Movable_objects[1]]
        # elif v_pred[1] == -1:
        #     movable_objects = [Movable_objects[0]]
        # else:
        #     movable_objects = Movable_objects
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
                    delta_y = 5.0
                    xmin = M_planning.x_min
                    xmax = M_planning.x_max
                    if v_pred[1] == 1:
                        ymin = M_planning.y_max - delta_y
                        ymax = M_planning.y_max + delta_y
                    else:
                        ymin = M_planning.y_min - delta_y
                        ymax = M_planning.y_min + delta_y
                if abs(v_pred[0]) == 1:
                    delta_x = 5.0
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
                # if num == 0:
                #     particles, weights = M_planning.generate_particles_in_polygon(particle_range, N=2000)
                # else:
                #     particles, weights = M_planning.update_particles(particles, weights, particle_range,
                #                                                      shift_3=Pos_list[-1]-Pos_list[0])
                particles, weights, particles_remain = M_planning.update_particles(particles, weights, particle_range,
                                                                                   Z_obs,
                                                                                   shift_3=Pos_list[-1] - Pos_list[0])
                weights = M_planning.weights_updates_dis(action=v_pred, Movable_objects_1=Movable_objects,
                                                         z_obs_list=Z_obs,
                                                         Pos_list_1=Pos_list, particles_5=particles, weights_5=weights)
                # 更新粒子，筛掉低权重粒子
                w_thresh = 1.0 / (len(particles) * 2)
                mask = weights >= w_thresh
                particles_filter = particles[mask]
                weights_filter = weights[mask]
                std = np.std(particles_filter, axis=0)
                if std[0] < 1.0 and std[1] < 1.0:
                    N = 4  ## 当粒子的范围足够小时，进入装配阶段
                else:
                    particles, weights = M_planning.resample_particles(particles_filter, weights_filter,
                                                                       particles_remain)
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
                    if abs(xmin - xmax)<0.5: xmin -= 1.5; xmax += 1.5
                    if abs(ymin - ymax)<0.5: ymin -= 1.5; ymax += 1.5  # 防止轮廓退化成一条直线
                    con_range = np.array([[xmin, ymin], [xmax, ymin],
                                          [xmax, ymax], [xmin, ymax]])
                    particle_range.append(con_range)
                # if num == 0:
                #     particles, weights = M_planning.generate_particles_in_polygon(particle_range, N=2000)
                # else:
                #     particles, weights = M_planning.update_particles(particles, weights, particle_range,
                #                                                      shift_3=Pos_list[-1] - Pos_list[0])
                particles, weights, particles_remain = M_planning.update_particles(particles, weights, particle_range, Z_obs,
                                                                 shift_3=Pos_list[-1] - Pos_list[0])
                weights = M_planning.weights_updates_dis(action=v_pred, Movable_objects_1=Movable_objects,
                                                         z_obs_list=Z_obs,
                                                         Pos_list_1=Pos_list, particles_5=particles, weights_5=weights)
                # 更新粒子，筛掉低权重粒子
                w_thresh = 1.0 / (len(particles) * 2)
                mask = weights >= w_thresh
                particles_filter = particles[mask]
                weights_filter = weights[mask]
                std = np.std(particles_filter, axis=0)
                if std[0] < 1.0 and std[1] < 1.0:
                    N = 4  ## 当粒子的范围足够小时，进入装配阶段
                else:
                    particles, weights = M_planning.resample_particles(particles_filter, weights_filter,
                                                                             particles_remain)


                ##更新图像
                scene_plot(scene, xs, ys, object_mask, obstacles, object_pos, ref_point, particles)
                z = 1


        # 更新权重
        # weights = M_planning.weights_updates(action=v_pred,
        #                                      objects=movable_objects, z_obs_list=Z_obs,
        #                                      particles=particles, weights=weights)

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
