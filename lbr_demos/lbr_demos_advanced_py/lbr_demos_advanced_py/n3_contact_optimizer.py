import time  # 导入计时模块：单调时钟用于状态超时，高精度时钟用于求解耗时统计。
from dataclasses import dataclass  # 导入数据类装饰器，为配置类自动生成初始化等方法。

import numpy as np  # 导入数组与线性代数库，约定 np 为 NumPy 的模块别名。
from scipy.optimize import minimize  # 导入数值最优化入口，后续以 SLSQP 求解三维位置增量。
from scipy.spatial.transform import Rotation  # 导入旋转表示工具，处理四元数、旋转矩阵与姿态差。


@dataclass  # 将配置类声明为数据类，允许调用者按字段覆盖默认参数。
class N3ContactConfig:  # 集中定义 N3 接触控制参数；默认值不等于调用方最终实际传入值。
    normal_min: float = 2.0  # 法向力软目标区间下限，单位 N，同时是低法向力线性惩罚的启动阈值。
    normal_max: float = 2.5  # 法向力软目标区间上限，单位 N；仅用于范围诊断和低法向力惩罚。
    force_reserve: float = 0.02  # 摩擦锥预留的力裕量，单位 N；判据为切向力加此裕量不超过安全摩擦力。
    mu_prior: float = 0.18  # 无标定情况下的摩擦系数先验；当前默认 0.18，不是真实静摩擦系数测量值。
    mu_factor: float = 0.7  # 保留的历史配置字段；本文件当前执行路径不读取该值。
    mu_floor: float = 0.05  # 摩擦估值初始化时的最低先验；不对后续每次 mu_safe 衰减施加此下限。
    mu_increase_rate: float = 0.05  # 保留的历史增长速率字段；当前采用 mu_safe_growth 的乘法更新。
    mu_max: float = 1.5  # 保留的历史固定上限字段；当前实际安全系数上限使用 mu_max_estimate。
    mu_safe_factor: float = 0.7  # 保留的兼容配置字段；N3 初始 mu_safe 由参考力比计算。
    mu_safe_growth: float = 1.1  # 摩擦恢复等待结束后，每次允许增长时将安全系数乘以 1.1。
    friction_recovery_timeout: float = 10.0  # 连续满足条件的位置恢复尝试时间，单位 s；满 10 s 才允许放宽摩擦系数。
    friction_penalty_start: float = 0.8  # 摩擦惩罚提前启动比例：预测切向力加裕量超过安全摩擦力的 80% 即惩罚。
    linear_speed: float = 0.0045  # 优化位置的线速度范数上限，单位 m/s；调用方可以覆盖该默认值。
    angular_speed: float = 0.03  # 保留的角速度配置，单位 rad/s；当前直接输出目标姿态，不用它积分限速。
    linear_acceleration: float = 0.01  # 优化线速度的逐轴变化限制，单位 m/s²；转为位置增量边界时乘 dt²。
    angular_acceleration: float = 0.1  # 保留的角加速度字段；当前没有独立角速度优化或积分控制。
    normal_recovery_gain: float = 0.0015  # 保留的历史法向恢复增益；当前位置恢复由目标函数优化完成。
    normal_recovery_speed: float = 0.001  # 保留的历史法向恢复速度字段；当前优化路径不读取。
    force_control_gain: float = 0.0035  # 保留的力控增益字段；实际 PID 公式在规划节点中实现，并非在此读取。
    object_linear_speed: float = 0.002  # 触觉标记质心位移阈值中的速度项，单位 m/s。
    object_angular_speed: float = np.deg2rad(3.0)  # 触觉物体角度变化阈值中的速度项，将 3 度/s 转成 rad/s。
    object_jump_distance: float = 0.00008  # 物体位移跳变基准阈值，单位 m，即 0.08 mm。
    object_jump_angle: float = np.deg2rad(0.12)  # 物体角度跳变基准阈值，将 0.12 度转成 rad。
    fit_residual: float = 0.0001  # 触觉点云刚体拟合均方根残差上限，单位 m，即 0.1 mm。
    position_tolerance: float = 0.0001  # 保留的位置容差字段；当前不跟踪原始位置目标，实际异常跟踪阈值另行写死。
    angle_tolerance: float = np.deg2rad(0.08)  # 真实姿态与当前目标姿态的允许误差，将 0.08 度转成 rad。
    correction_limit: float = 0.002  # 单周期位置增量范数的上限，单位 m；不是相对轨迹起点的累计偏移上限。
    sensor_timeout: float = 0.2  # 控制调用间隔和传感器年龄的超时阈值，单位 s，当前为 200 ms。
    progress_timeout: float = 10.0  # 没有推进轨迹点的超时阈值，单位 s；部分恢复/等待会调整对应计时。
    wrench_recovery_timeout: float = 10.0  # 极限力或接触丢失 PID 恢复的墙钟时间上限，单位 s，等待数据也不延长它。
    recovery_frames: int = 10  # 认定触觉物体稳定所需的连续新鲜稳定帧数。
    stiffness_floor: float = 100.0  # 刚度矩阵特征值下限，单位 N/m；亦用于缺失刚度时的各向同性回退。
    stiffness_ceiling: float = 100000.0  # 刚度矩阵特征值上限，单位 N/m，避免模型包含过大的响应尺度。
    objective_normal_weight: float = 4.0  # 法向力趋近参考法向力的平方误差权重。
    objective_normal_underforce_weight: float = 10.0  # 法向力低于 2 N 时的线性不足量惩罚权重；当前代码值为 10。
    objective_tangent_weight: float = 8.0  # 预测切向力模长平方的权重；当前代码值为 8。
    objective_friction_weight: float = 100.0  # 预测摩擦力接近或超过提前惩罚边界时的平方惩罚权重。
    objective_smooth_weight: float = 0.2  # 相邻线速度变化平方的权重，用于抑制位置指令速度突变。
    objective_object_weight: float = 50.0  # 预测力变化的基础惩罚权重，随后乘以 1+物体运动指标；当前代码值为 50。
    objective_wrench_weight: float = 0.5  # 上一实测力变化与本次预测力变化之和的平方惩罚权重。
    optimization_iterations: int = 30  # SLSQP 单次求解最多允许的迭代次数。
    optimization_tolerance: float = 1.0e-9  # SLSQP 的函数收敛容差；不是以米表示的位置精度。
    optimization_budget_ms: float = 6.0  # 单次位置优化的耗时预算，单位 ms；超预算候选不会用于发布。
    control_period: float = 0.01  # 名义控制周期，单位 s，对应 100 Hz，但不保证 ROS 实际调度达到该频率。


class N3ContactOptimizer:  # 定义有状态的 N3 单步控制器，输出位置、姿态和是否允许推进轨迹。
    (  # 用相邻字符串保留原类文档原文，方便逐行添加中文注释。
        "One-step model-assisted N3 contact controller.\n"  # 类文档：使用刚度模型辅助的 N3 单步接触控制器。
        "\n"  # 保留原文档中的空行，文档字符串内容不变。
        "    The measured wrench is used directly for hard safety gates.  A stiffness\n"  # 原文说明：实测 wrench 用于硬安全判断，刚度来自主动探索。
        "    matrix from the active probe predicts the force change caused by a small\n"  # 原文说明：用刚度矩阵预测小幅位置增量引起的力变化。
        "    position increment, and a constrained SLSQP solver minimizes the current\n"  # 原文说明：受约束 SLSQP 最小化当前接触目标函数。
        "    contact objective before the command position is published.\n"  # 原文说明：求解完成后再由调用节点发布位置指令。
        "    "  # 保留原文档结尾的缩进字符，确保类文档字符串完全一致。
    )  # 结束类文档字符串；相邻字符串由 Python 自动合并。

    def __init__(self, position, quat, clouds, config=None, force_bias=None,  # 初始化控制器：传入当前位姿、双侧触觉点云、配置及初始力偏置。
                 normal=None, stiffness=None, normal_force_reference=None,  # 继续列出法向、刚度和法向参考力输入。
                 tangent_force_reference=None, mu_safe = None):  # 接收 N3 开始前稳定接触窗口得到的切向参考力。
        self.config = config or N3ContactConfig()  # 优先采用调用者配置，否则创建默认配置对象。
        self.command_position = np.asarray(position, dtype=float).copy()  # 保存初始位置指令的浮点副本，避免与调用方数组共享可变内存。
        quat = np.asarray(quat, dtype=float)  # 将初始四元数转换为浮点数组，约定分量顺序为 x、y、z、w。
        if quat.shape != (4,) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-9:  # 检查四元数为四维有限向量且模长非零。
            raise ValueError('N3 requires a valid initial quaternion.')  # 初始四元数无效时拒绝创建控制器。
        self.command_rotation = Rotation.from_quat(quat)  # 由初始四元数建立当前指令旋转，Rotation 会处理四元数归一化。
        self.force_reference_rotation = self.command_rotation  # 保存创建控制器时的参考旋转，用于后续力坐标变换。
        self.force_bias = np.zeros(3) if force_bias is None else np.asarray(force_bias, dtype=float).copy()  # 未提供偏置时使用零向量，否则复制调用者给出的三维力偏置。
        if self.force_bias.shape != (3,) or not np.all(np.isfinite(self.force_bias)):  # 校验力偏置必须是三个有限实数。
            raise ValueError('N3 requires a finite force bias.')  # 偏置格式错误时抛出异常，由调用方决定如何处理。
        normal = np.asarray(normal, dtype=float)  # 将估计接触法向转换为浮点数组。
        normal_norm = np.linalg.norm(normal)  # 计算法向长度，为有效性判断及归一化作准备。
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or normal_norm < 1e-9:  # 法向必须为三维有限非零向量；不会默认替换为 Z 轴。
            raise ValueError('N3 requires a valid estimated_contact_normal.')  # 无效法向直接使初始化失败，防止错误的力投影。
        self.normal = normal / normal_norm  # 保存单位法向，初始化后本文件不会在线重新辨识此向量。
        self.tangent_projector = np.eye(3) - np.outer(self.normal, self.normal)  # 构造切平面投影矩阵 I−nnᵀ，用于提取完整二维切向分量。
        self.reference_clouds = [np.array(cloud, dtype=float, copy=True) for cloud in clouds]  # 复制双侧参考触觉标记点云，作为后续刚体拟合的固定参考。
        if len(self.reference_clouds) != 2:  # 要求恰好有两侧点云，对应左右触觉传感器。
            raise ValueError('N3 requires two tactile marker clouds.')  # 点云侧数不匹配时拒绝初始化。
        for cloud in self.reference_clouds:  # 逐侧验证参考点云。
            self._fit_cloud(cloud, cloud)  # 用点云与自身拟合检查其形状、有限性和几何可辨识性。

        self.previous_objects = None  # 上一组物体拟合结果为空，首个新鲜观测尚无相邻帧差分。
        self.object_stamp = None  # 上一组物体观测处理时刻，使用单调时钟。
        self.sensor_stamps = None  # 上一组双侧点云的输入时间戳，用于判断是否是真正的新数据。
        self.stable_frames = 0  # 初始化连续稳定帧计数。
        self.object_unstable = False  # 初始化物体不稳定状态标志。
        self.normal_force_reference = float(  # 保存 N3 偏转使用的法向参考力。
            0.5 * (self.config.normal_min + self.config.normal_max)
            if normal_force_reference is None else normal_force_reference)
        self.tangent_force_reference = float(  # 保存 N3 偏转使用的切向参考力。
            0.0 if tangent_force_reference is None else tangent_force_reference)
        reference_ratio = max(self.tangent_force_reference, 0.0) / max(  # 根据参考力比计算初始摩擦安全系数。
            abs(self.normal_force_reference), 1.0e-9)
        self.mu_safe = mu_safe if mu_safe is not None else 1.2 * reference_ratio  # 初始安全摩擦系数为参考切向/法向力比的 1.2 倍。
        self.mu_max_estimate = max(  # 保证后续的最大估值上限不会立即截断参考值初始化结果。
            self.config.mu_prior, self.config.mu_floor, self.mu_safe)
        self.force_ratios = []  # 存储最近五次合格新鲜观测的切向/法向力比。
        self.stiffness = self._prepare_stiffness(stiffness)  # 对输入刚度进行对称化、特征值限幅或回退处理。
        self.previous_velocity = np.zeros(6)  # 保存上一次六维指令速度；前三维为线速度，后三维当前保持零。
        self.previous_time = None  # 尚无上一周期调用时刻，首次 step 会用名义周期初始化。
        self.last_progress = None  # 尚无最后一次轨迹推进时刻，用于后续无进展保护。
        self.control_gap_wait_started_at = None  # 控制调用间隔超时时的等待起点，空值表示未等待。
        self.control_gap_recovering = False  # 控制间隔恢复后的姿态门控标志，防止立即跟随新目标姿态。
        self.friction_recovery_started_at = None  # 摩擦位置恢复的连续计时起点。
        self.friction_hold_rotation = None  # 摩擦越界时锁定的实测旋转，避免继续偏转。
        self.fault = None  # 锁存故障原因；一旦非空，正常 step 首先返回保持。
        self.wrench_recovery_started_at = None  # 极限力 PID 恢复开始时刻，计时不随重复触发重置。
        self.wrench_recovery_rotation = None  # 极限力恢复时固定的实测姿态。
        self.failures = 0  # 保留的失败计数字段；当前本文件没有递增或读取该计数。
        self.status = {}  # 控制状态字典，由调用节点读取并输出日志。
        self.previous_force_world = None  # 上一新鲜触觉帧对应的参考坐标系力向量。
        self.object_motion = 0.0  # 缓存无量纲物体运动指标，非新鲜帧继续使用上次结果。
        self.object_wrench_change = 0.0  # 缓存相邻新鲜观测的力变化模长。
        self.force_change = np.zeros(3)  # 缓存实际三维力差，不能用一个模长替代向量方向。
        self.optimization_status = {}  # 保存求解器结果、迭代次数、耗时和约束余量等信息。

    def _prepare_stiffness(self, stiffness):  # 将输入刚度整理为有限、对称、特征值有界的三维矩阵。
        config = self.config  # 读取配置，避免重复访问实例属性。
        if stiffness is None:  # 判断调用者是否未提供刚度。
            matrix = np.eye(3) * config.stiffness_floor  # 缺少刚度时使用下限乘单位阵作为模型假设。
        else:  # 处理调用者已提供刚度的情况。
            matrix = np.asarray(stiffness, dtype=float)  # 将输入刚度转换为浮点数组。
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):  # 检查刚度是否为有限的 3×3 矩阵。
                matrix = np.eye(3) * config.stiffness_floor  # 格式不合格时也使用各向同性下限刚度，不中断初始化。
        matrix = 0.5 * (matrix + matrix.T)  # 取矩阵与转置的平均，舍弃反对称部分。
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)  # 对实对称矩阵求升序特征值及对应的正交特征向量。
        eigenvalues = np.clip(eigenvalues, config.stiffness_floor,  # 将每个特征值至少提高到刚度下限。
                              config.stiffness_ceiling)  # 同时将每个特征值限制在刚度上限以内。
        return (eigenvectors * eigenvalues) @ eigenvectors.T  # 按特征向量重建对称正定刚度；非对角耦合通常仍然保留。

    @staticmethod  # 声明静态工具函数，不需要访问控制器实例。
    def _skew(vector):  # 构造三维向量的叉乘矩阵；当前执行路径未调用此工具。
        return np.array([[0.0, -vector[2], vector[1]],  # 叉乘矩阵第一行，使矩阵乘另一向量等价于叉乘。
                         [vector[2], 0.0, -vector[0]],  # 叉乘矩阵第二行，保留对应分量的正负号。
                         [-vector[1], vector[0], 0.0]])  # 叉乘矩阵第三行，并返回完整 3×3 数组。

    @staticmethod  # 点云拟合只依赖参数，声明为静态方法。
    def _fit_cloud(reference, current):  # 拟合参考点云到当前点云的刚体旋转，返回质心、旋转和残差。
        reference = np.asarray(reference, dtype=float)  # 将参考点云转换为浮点数组。
        current = np.asarray(current, dtype=float)  # 将当前点云转换为浮点数组。
        if (reference.shape != current.shape or current.ndim != 2  # 检查两组点云形状一致，且当前点云是二维数组。
                or current.shape[1] != 3 or len(current) < 3  # 要求每点三个坐标，点数至少为三个。
                or not np.all(np.isfinite(current))):  # 要求当前点云所有坐标均为有限值。
            raise ValueError('Invalid N3 tactile markers.')  # 点云不满足基本条件时抛出异常。
        reference_center = reference.mean(axis=0)  # 计算参考点云的三维质心。
        current_center = current.mean(axis=0)  # 计算当前点云的三维质心；它是后续相邻帧位移的代理量。
        reference_centered = reference - reference_center  # 参考点云去中心化，消除平移对旋转拟合的影响。
        current_centered = current - current_center  # 当前点云去中心化，使用同样的处理。
        left, singular, right = np.linalg.svd(reference_centered.T @ current_centered)  # 对两组中心化点云的互协方差矩阵进行奇异值分解。
        if singular[1] < 1e-12:  # 第二奇异值过小表示几何接近共线或退化，旋转估计不可靠。
            raise ValueError('Degenerate N3 tactile marker geometry.')  # 拒绝退化点云，避免产生任意或突变的拟合旋转。
        correction = np.eye(3)  # 建立反射修正矩阵，默认不改变坐标轴。
        correction[2, 2] = np.linalg.det(right.T @ left.T)  # 按候选旋转的行列式修正第三个方向，得到真旋转而非镜像。
        matrix = right.T @ correction @ left.T  # 组合 SVD 因子与修正矩阵得到参考到当前的旋转矩阵。
        residual = np.sqrt(np.mean(np.sum(  # 开始计算旋转后参考点与当前点之间的均方根欧氏残差。
            (reference_centered @ matrix.T - current_centered) ** 2, axis=1)))  # 逐点累加三轴误差平方，再对点平均并开平方；单位与点云相同。
        return current_center, Rotation.from_matrix(matrix), residual  # 返回当前质心、Rotation 对象及拟合残差。

    def start_segment(self):  # 开始新轨迹段时重置部分历史；不会清空 mu_safe、最大力比或锁存故障。
        self.previous_time = None  # 清除上一调用时间，使下次 step 按名义周期起步。
        self.last_progress = None  # 清除最后推进时刻，下次 step 重新开始无进展计时。
        self.control_gap_wait_started_at = None  # 结束先前控制间隔异常的数据等待标记。
        self.control_gap_recovering = False  # 清除控制间隔恢复阶段的姿态门控标志。
        self.friction_recovery_started_at = None  # 重新开始摩擦位置恢复计时。
        self.friction_hold_rotation = None  # 清除上段摩擦越界锁定姿态。
        self.previous_velocity[:] = 0.0  # 清零历史指令速度，下一段从零历史速度优化。
        self.stable_frames = 0  # 新段重新累计稳定帧。
        self.force_ratios = []  # 清空五帧力比窗口，但保留累计最大估值。
        self.previous_force_world = None  # 重建力差历史，不将跨段力差算作单周期变化。
        self.object_motion = 0.0  # 清零缓存的物体运动指标。
        self.object_wrench_change = 0.0  # 清零缓存的力变化模长。
        self.force_change[:] = 0.0  # 原位清零力差向量。

    def _hold(self, reason, latch=False, mode='hold'):  # 统一返回保存位姿并禁止推进；mode 可为 hold、recovery 或 waiting_data。
        if latch:  # 仅当调用者显式要求锁存时保存故障。
            self.fault = reason  # 保存故障原因，后续 step 首先拦截该故障。
        self.previous_velocity[:] = 0.0  # 清零历史六维指令速度；不代表实际机械臂速度瞬间变为零。
        self.status.update(mode=mode, reason=reason, mu_safe=self.mu_safe,  # 更新显示模式、原因和当前安全摩擦系数。
                           tracking_allowed=False, advance=False,  # 明确禁止跟踪推进；该分支不会更新轨迹索引。
                           command_velocity=self.previous_velocity.copy(),  # 将零速度副本写入状态字典，避免后续共享修改。
                           linear_speed=0.0, angular_speed=0.0)  # 更新标量速度日志，均为零。
        return self.command_position.copy(), self.command_rotation.as_quat(), False  # 返回保存的位置、保存的姿态四元数和 False；本函数本身不发布 ROS 消息。

    def _update_friction_estimate(self, tangent_force, normal_force, object_dt,  # 按合格力观测更新摩擦估计；object_dt 参数保留但当前未使用。
                                  stable_ok, object_jump, wrench_range_ok,  # 接收物体稳定性、跳变与力/力矩范围判据。
                                  friction_ok, fresh, allow_update=True,  # 接收当前摩擦判据、数据新鲜度和是否允许更新。
                                  allow_increase=False):  # 单独接收是否已完成等待、允许增大安全摩擦系数的门控。
        if not (allow_update and fresh and stable_ok and not object_jump  # 只有允许更新、新鲜、稳定且没有跳变时才继续。
                and wrench_range_ok):  # 还要求总力/力矩处于调用方认可的范围。
            return  # 任何条件不满足时保留原估计，不更新五帧窗口。
        ratio = tangent_force / max(abs(normal_force), 1e-9)  # 计算切向力模长除以法向力绝对值，用小量防止除零。
        if np.isfinite(ratio):  # 只接受有限力比。
            self.force_ratios = (self.force_ratios + [float(ratio)])[-5:]  # 追加本帧力比，仅保存最近五个结果供日志均值使用。
            self.mu_max_estimate = max(self.mu_max_estimate, float(ratio))  # 更新历史最大观测力比；它不是材料极限静摩擦系数的直接测量。
        if not friction_ok and allow_increase:  # 只有当前摩擦条件不满足且允许增加时，才放宽安全系数。
            self.mu_safe = min(  # 安全系数取增长值和当前最大力比估值中的较小者。
                self.mu_max_estimate,  # 增长上限是在线最大力比估值，而非配置字段 mu_max。
                self.config.mu_safe_growth * self.mu_safe)  # 按乘法因子增加安全系数；正常跟踪不会在此持续增大。

    def _objective_and_gradient(self, delta, force_world, dt, object_motion,  # 计算以米为变量的位置增量目标函数及其三维解析梯度。
                                wrench_change, mu_safe):  # 接收上一实测力差和安全摩擦系数，用于连续受力及摩擦惩罚。
        config = self.config  # 读取本次目标函数使用的配置。
        delta_force = -self.stiffness @ delta  # 使用固定刚度模型 Δf=−KΔp 预测位置变化产生的力增量。
        predicted_force = force_world + delta_force  # 将预测力增量叠加到当前参考坐标系实测力上。
        normal_force = float(np.dot(self.normal, predicted_force))  # 将预测合力投影到单位法向，得到有符号法向力。
        tangent_vector = self.tangent_projector @ predicted_force  # 将预测合力投影到切平面，得到三维表示的切向向量。
        tangent_force = float(np.linalg.norm(tangent_vector))  # 计算切向力模长，单位 N，非单独 X 或 Y 分量。
        normal_gradient = -self.stiffness.T @ self.normal  # 预测法向力对位置增量的梯度为 −Kᵀn。
        tangent_gradient = -self.stiffness.T @ tangent_vector / max(tangent_force, 1.0e-12)  # 计算预测切向模长梯度，零模长时用微小分母保护。
        normal_error = normal_force - self.normal_force_reference  # 法向力与 N3 稳定接触参考值的误差。
        penalty_mu = config.friction_penalty_start * mu_safe  # 提前惩罚阈值采用安全摩擦系数的 80%，硬摩擦锥仍用完整系数。
        friction_violation = max(0.0, tangent_force + config.force_reserve  # 计算切向力加裕量超过提前边界的正部。
                                  - penalty_mu * max(normal_force, 0.0))  # 负法向力不提供摩擦容量；提前边界以内的违规量取零。
        velocity = delta / dt  # 位置增量除以时间步，得到候选线速度。
        velocity_change = velocity - self.previous_velocity[:3]  # 计算候选线速度相对于上一指令线速度的变化。
        stability_weight = config.objective_object_weight * (1.0 + object_motion)  # 物体运动越大，越强烈惩罚预测力变化；这是稳定性的代理目标。
        total_force_change = wrench_change + delta_force  # 上一实测力差加本次模型预测力差，保留三维方向。
        cost = (  # 开始累加七项标量代价。
            config.objective_normal_weight * normal_error ** 2  # 法向力目标的平方误差惩罚。
            + config.objective_normal_underforce_weight * max(0.0, self.normal_force_reference - normal_force)  # 低于 normal_min 的不足量线性惩罚，到达下限即为零。
            + config.objective_tangent_weight * (tangent_force - self.tangent_force_reference) ** 2  # 切向力趋近 N3 稳定接触参考值。
            + config.objective_friction_weight * friction_violation ** 2  # 提前摩擦边界违规量平方惩罚，权重默认较大。
            + config.objective_smooth_weight * float(np.dot(velocity_change, velocity_change))  # 候选线速度变化的平方范数，抑制速度变化。
            + stability_weight * float(np.dot(delta_force, delta_force))  # 预测力增量的平方范数，按当前物体运动指标加权。
            + config.objective_wrench_weight  # 引入相邻受力连续性项的权重。
            * float(np.dot(total_force_change, total_force_change))  # 对上一实测力差与预测力差之和作平方惩罚。
        )  # 结束目标函数求和，代价含不同物理量的加权组合。
        gradient = (  # 开始累加目标函数对三维位置增量的解析梯度。
            2.0 * config.objective_normal_weight * normal_error * normal_gradient  # 法向力平方项用链式法则得到 2w·误差·法向梯度。
            - config.objective_normal_underforce_weight * normal_gradient * (normal_force < config.normal_min)  # 不足量线性项仅低于阈值时有梯度，恰在阈值取零次梯度。
            + 2.0 * config.objective_tangent_weight * (tangent_force - self.tangent_force_reference) * tangent_gradient  # 切向参考力误差的链式梯度。
            + 2.0 * config.objective_friction_weight * friction_violation  # 摩擦平方惩罚的链式系数为两倍权重乘违规量。
            * (tangent_gradient - penalty_mu * normal_gradient * (normal_force > 0.0))  # 再乘切向模长梯度减提前摩擦容量梯度；正法向时才对容量求导。
            + 2.0 * config.objective_smooth_weight * velocity_change / dt  # 速度平滑项对位置求导还需除以时间步。
            - 2.0 * stability_weight * self.stiffness.T @ delta_force  # 预测力变化平方项使用 Δf 对位置的导数 −K。
            - 2.0 * config.objective_wrench_weight * self.stiffness.T @ total_force_change  # 连续受力项使用相同力增量导数，保留已有实测力差方向。
        )  # 结束三维梯度求和。
        return float(cost), gradient  # 返回标量代价和三维梯度数组。

    def _objective(self, delta, force_world, dt, object_motion,  # 只需代价值时使用此包装函数。
                   wrench_change, mu_safe):  # 参数与目标及梯度函数一致，避免重复实现公式。
        return self._objective_and_gradient(  # 调用完整目标和梯度计算。
            delta, force_world, dt, object_motion, wrench_change, mu_safe)[0]  # 仅返回结果的第零项，即标量目标值。

    def _optimize_position(self, position, force_world, dt, object_motion,  # 求解下一位置；position 是当前实测位置，非原始轨迹位置。
                           wrench_change, mu_safe, enforce_cone=True):  # enforce_cone 控制是否同时强制预测法向下限与摩擦锥。
        """Solve for p_next = position + scale * solution using SLSQP."""  # 原始文档字符串：用 SLSQP 求下一位置等于当前位置加尺度化解。
        config = self.config  # 读取求解参数。
        started_at = time.perf_counter()  # 记录高精度开始时刻，预算包含构建问题和执行求解。
        scale = min(config.correction_limit, config.linear_speed * dt)  # 取单步位移上限与最大速度乘 dt 的较小者作为变量尺度。
        previous_delta = self.previous_velocity[:3] * dt  # 把上一线速度转为当前时间步下的参考位置增量。
        acceleration_delta = config.linear_acceleration * dt * dt  # 逐轴速度变化上限转为位移增量差上限 a·dt²。
        lower = np.maximum(-1.0, (previous_delta - acceleration_delta) / scale)  # 归一化变量下界同时受负尺度边界和逐轴减速度限制。
        upper = np.minimum(1.0, (previous_delta + acceleration_delta) / scale)  # 归一化变量上界同时受正尺度边界和逐轴加速度限制。
        initial = np.clip(previous_delta / scale, lower, upper)  # 以上一速度对应的增量为初值，并截断到当前上下界。
        force_matrix = self.stiffness * scale  # 将刚度乘位置尺度，使后续 candidate 为无量纲变量。
        normal_matrix = self.normal @ force_matrix  # 计算归一化位置变量对法向力变化的线性系数，符号在预测式中取负。
        tangent_matrix = self.tangent_projector @ force_matrix  # 计算归一化位置变量对切向向量变化的线性矩阵。
        normal_error = self.normal @ force_world - self.normal_force_reference  # 当前法向力相对 N3 稳定接触参考值的误差。
        tangent_force = self.tangent_projector @ force_world  # 当前切向力向量；此处变量虽名为 force，保存的并非模长。
        tangent_norm = np.linalg.norm(tangent_force)  # 计算当前切向力模长，用于二次热启动近似。
        tangent_unit = tangent_force / max(tangent_norm, 1.0e-12)  # 计算切向力方向，避免参考目标项的线性化除零。
        stability_weight = config.objective_object_weight * (1.0 + object_motion)  # 当前周期的物体稳定代理权重。
        hessian = 2.0 * (  # 构造所有光滑二次项的 Hessian；线性不足量和摩擦正部项另算。
            config.objective_normal_weight * np.outer(normal_matrix, normal_matrix)  # 法向力平方项对归一化变量的二阶矩阵。
            + config.objective_tangent_weight * tangent_matrix.T @ tangent_matrix  # 切向力平方项对归一化变量的二阶矩阵。
            + config.objective_smooth_weight * (scale / dt) ** 2 * np.eye(3)  # 速度平滑项的对角二阶矩阵。
            + (stability_weight + config.objective_wrench_weight) * force_matrix.T @ force_matrix)  # 两个力变化平方项的二阶矩阵，合并共享的刚度乘积。
        linear = -2.0 * (  # 构造二次目标关于归一化变量的一次系数。
            config.objective_normal_weight * normal_error * normal_matrix  # 法向力误差产生的一次项。
            + config.objective_tangent_weight * tangent_matrix.T @ tangent_force  # 当前切向向量平方项产生的一次项。
            - config.objective_tangent_weight * self.tangent_force_reference * tangent_matrix.T @ tangent_unit  # 切向参考值产生的一次近似项。
            + config.objective_smooth_weight * scale / dt * self.previous_velocity[:3]  # 上一指令速度产生的平滑一次项。
            + config.objective_wrench_weight * force_matrix.T @ wrench_change)  # 上一实测力差产生的连续受力一次项。
        constant = (  # 构造候选增量为零时的二次目标常数部分。
            config.objective_normal_weight * normal_error ** 2  # 当前法向误差的平方代价。
            + config.objective_tangent_weight * (tangent_norm - self.tangent_force_reference) ** 2  # 当前切向参考力误差的代价。
            + config.objective_smooth_weight * (self.previous_velocity[:3] @ self.previous_velocity[:3])  # 候选速度为零时相对上一指令速度的平滑代价。
            + config.objective_wrench_weight * (wrench_change @ wrench_change))  # 候选力增量为零时的相邻实测力差代价。
        penalty_mu = config.friction_penalty_start * mu_safe  # 计算本周期提前摩擦惩罚采用的有效系数。

        # def objective(candidate):  # 定义供 SLSQP 调用的精确目标及解析梯度回调。
        #     delta = scale * np.asarray(candidate, dtype=float)  # 将无量纲候选变量转换为米制位置增量。
        #     value, gradient = self._objective_and_gradient(  # 使用与诊断完全一致的参考力目标函数。
        #         delta, force_world, dt, object_motion, wrench_change, mu_safe)
        #     return float(value), scale * gradient  # 将米制增量梯度转换为无量纲候选变量梯度。

        def objective(candidate):  # 定义供 SLSQP 调用的标量目标及解析梯度回调。
            predicted = force_world - force_matrix @ candidate  # 归一化增量乘已缩放刚度，预测下一步合力。
            normal_force = self.normal @ predicted  # 计算预测法向力。
            tangent = self.tangent_projector @ predicted  # 计算预测切向力向量。
            tangent_norm = np.linalg.norm(tangent)  # 计算预测切向力模长。
            violation = max(0.0, tangent_norm + config.force_reserve  # 求带 0.02 N 裕量的切向力与提前边界之差的正部。
                            - penalty_mu * max(normal_force, 0.0))  # 仅正法向提供摩擦容量，边界内此项为零。
            product = hessian @ candidate  # 计算 Hessian 与候选变量的乘积，后续代价和梯度共用。
            value = constant + linear @ candidate + 0.5 * candidate @ product  # 按常数项、一次项和二次项得到基础代价。
            gradient = linear + product  # 基础二次目标的梯度为一次系数加 Hessian 乘变量。
            if normal_force < self.normal_force_reference:  # 预测法向力低于软目标下限时启用线性不足惩罚。
                value += config.objective_normal_underforce_weight * (self.normal_force_reference - normal_force)  # 按法向力不足量直接增加代价，不取平方。
                gradient += config.objective_normal_underforce_weight * normal_matrix  # 补上不足量对归一化位置变量的解析梯度。
            if violation > 0.0:  # 接近摩擦边界的正部大于零时，额外加入摩擦惩罚。
                violation_gradient = (-tangent @ force_matrix / max(tangent_norm, 1.0e-12)  # 计算切向模长随归一化位置变化的导数。
                                        + penalty_mu * normal_matrix * (normal_force > 0.0))  # 加上被减去的提前法向摩擦容量对应的导数。
                value += config.objective_friction_weight * violation ** 2  # 增加摩擦违规量平方的加权代价。
                gradient += 2.0 * config.objective_friction_weight * violation * violation_gradient  # 按链式法则叠加摩擦平方惩罚的梯度。
            return float(value), gradient  # 返回标量代价与归一化三维变量梯度，jac=True 将使用它。


        def constraints(candidate):  # 定义 SLSQP 不等式，约定返回值每项必须大于等于零。
            predicted = force_world - force_matrix @ candidate  # 按当前候选增量预测合力。
            normal_force = self.normal @ predicted  # 计算预测法向力。
            tangent_force = np.linalg.norm(self.tangent_projector @ predicted)  # 计算预测切向力模长。
            values = [1.0 - candidate @ candidate]  # 范数约束 1−zᵀz≥0，保证总位移不超过单步尺度。
            if enforce_cone:  # 只有正常跟踪启用预测接触/摩擦硬约束。
                values.extend([normal_force - 0.1,  # 加入预测法向力至少为 0.1 N 的余量。
                               mu_safe * normal_force - tangent_force - config.force_reserve])  # 加入完整 mu_safe 摩擦锥余量，并扣除力裕量。
            return np.asarray(values)  # 返回一维不等式余量数组；恢复时通常只有位移范数约束。

        def constraint_jacobian(candidate):  # 为不等式提供解析雅可比，避免逐项有限差分。
            predicted = force_world - force_matrix @ candidate  # 重算候选增量对应的预测力。
            tangent = self.tangent_projector @ predicted  # 取预测切向向量。
            normal_gradient = -normal_matrix  # 预测法向力对归一化变量的梯度为负法向系数。
            rows = [-2.0 * candidate]  # 范数不等式对三维变量的梯度是 −2z。
            if enforce_cone:  # 仅当对应硬约束启用时才补充其梯度行。
                tangent_gradient = -tangent @ force_matrix / max(np.linalg.norm(tangent), 1.0e-12)  # 计算切向模长梯度，避免零切向力时除零。
                rows.extend([normal_gradient, mu_safe * normal_gradient - tangent_gradient])  # 依次加入法向下限和摩擦锥不等式的梯度行。
            return np.asarray(rows)  # 返回约束条数乘三维变量数的雅可比矩阵。

        def check_budget(candidate):  # 每次迭代回调检查累计求解预算；candidate 为接口要求的未使用参数。
            if (time.perf_counter() - started_at) * 1000.0 > config.optimization_budget_ms:  # 高精度耗时转为毫秒后与预算比较。
                raise TimeoutError('optimization_budget_exceeded')  # 超预算时抛出 TimeoutError，外层捕获并拒绝候选。

        initial_cost = objective(initial)[0]  # 计算原始热启动初值的完整代价，供日志比较。
        try:  # 尝试用二次部分的无约束最优点改善初值。
            quadratic_initial = np.clip(np.linalg.solve(hessian, -linear), lower, upper)  # 解 H·z=−linear，再逐轴裁剪到允许范围。
            if (np.min(constraints(quadratic_initial)) >= 0.0  # 只有候选满足所有当前启用的不等式时才允许替换。
                    and objective(quadratic_initial)[0] < initial_cost):  # 候选的完整目标还必须优于原热启动初值。
                initial = quadratic_initial  # 采用更好的初值；仍然需要后续 SLSQP 完整求解。
        except np.linalg.LinAlgError:  # 二次矩阵不可逆等线性代数错误不阻止使用原初值。
            pass  # 忽略初值改进失败，继续执行正式优化。
        self.optimization_status = dict(solver_success=False, solver_iterations=0)  # 初始化本次求解状态为失败、零迭代，避免沿用旧结果。
        try:  # 开始可捕获预算和数值异常的正式求解。
            result = minimize(  # 调用 SciPy 通用优化接口。
                objective, initial, jac=True, method='SLSQP',  # 设置目标、初始变量、解析梯度接口和 SLSQP 算法。
                bounds=list(zip(lower, upper)),  # 传入三个分量的上下界，约束速度变化。
                constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jacobian},  # 传入不等式余量及其雅可比。
                callback=check_budget,  # 每次迭代后检查求解时间预算；不是操作系统级抢占中断。
                options={'maxiter': config.optimization_iterations,  # 设置最大迭代次数。
                         'ftol': config.optimization_tolerance})  # 设置目标收敛容差，并完成求解器调用。
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0  # 求解结束后重新计算总耗时，包含未被迭代回调即时捕获的时间。
            feasible = (np.all(np.isfinite(result.x))  # 开始独立复核解是否有限，不能只依赖求解器成功标志。
                        and np.all(result.x >= lower - 1.0e-8)  # 检查所有分量不低于下界，允许 1e−8 数值误差。
                        and np.all(result.x <= upper + 1.0e-8)  # 检查所有分量不高于上界。
                        and np.min(constraints(result.x)) >= -1.0e-8)  # 检查全部不等式余量不少于 −1e−8。
            success = bool(result.success and feasible and elapsed_ms <= config.optimization_budget_ms)  # 必须收敛、可行且未超预算，才认可本次解。
            self.optimization_status.update(  # 写入本次求解诊断字典。
                solver_success=success, solver_iterations=int(result.nit),  # 记录综合成功状态与迭代次数。
                solver_message=('optimization_budget_exceeded' if elapsed_ms > config.optimization_budget_ms  # 若耗时超预算，用明确超时原因覆盖求解器消息。
                                else str(result.message)), solver_ms=elapsed_ms,  # 否则保留求解器原始消息，并记录耗时。
                predicted_constraints_enforced=bool(enforce_cone),  # 标明本次是否启用了预测摩擦锥及法向下限约束。
                solver_constraint_margin=float(np.min(constraints(result.x))),  # 记录所有启用约束中的最小余量，不专指摩擦锥。
                solver_budget_exceeded=elapsed_ms > config.optimization_budget_ms)  # 记录求解预算是否超出。
            if not success:  # 任一成功条件不满足时拒绝输出候选位置。
                return None, np.zeros(3), initial_cost, float(result.fun)  # 返回空位置、零增量和代价值，由 step 进入可重试保持。
            delta = result.x * scale  # 将求解得到的无量纲变量转换回米制位置增量。
            final_cost = float(result.fun)  # 记录最终完整目标函数值。
        except (TimeoutError, ValueError, FloatingPointError) as error:  # 捕获预算超时、参数错误及显式浮点异常。
            self.optimization_status.update(  # 为异常路径补充求解状态。
                solver_message=str(error),  # 记录异常文本供日志诊断。
                solver_ms=1000.0 * (time.perf_counter() - started_at),  # 即使异常也计算本次已消耗的时间。
                solver_budget_exceeded=isinstance(error, TimeoutError))  # 仅 TimeoutError 在此直接标为预算超出。
            return None, np.zeros(3), initial_cost, initial_cost  # 异常路径不输出可执行位置，代价回退为原初值代价。
        return position + delta, delta, initial_cost, final_cost  # 成功时返回下一绝对位置、米制增量、原初值代价及最终代价。

    def step(self, position, quat, force, moment, target_position, target_quat,  # 执行一次控制计算：输入当前位姿、测量力矩以及原始目标位姿。
             pivot, clouds, sensor_times, contact_active, now=None, final_point=False,  # 接收接触参考点、双侧点云、传感器时间戳和接触状态；可指定时刻及末端路径点标志。
             force_control=False, wrench_in_target=None):  # 接收外部力控制状态及外部范围判断，返回位置指令、姿态指令和轨迹推进许可。
        started_at = time.perf_counter()  # 记录本次控制调用的高精度起始时间，用于统计运行耗时。
        now = time.monotonic() if now is None else now  # 未指定当前时刻时使用单调时钟，避免系统时间校正影响超时判断。
        config = self.config  # 取得当前配置对象，便于后续读取阈值和权重。
        previous_mode = self.status.get('mode')  # 保存上次控制模式，用于检测从跟踪到不稳定状态的切换。
        if self.fault is not None:  # 检查是否已有锁存故障。
            return self._hold(self.fault)  # 存在锁存故障时保持现有指令并禁止推进，不再执行本轮优化。
        if (self.wrench_recovery_started_at is not None  # 若已经进入力/力矩 PID 恢复，则检查恢复持续时间。
                and now - self.wrench_recovery_started_at >= config.wrench_recovery_timeout):  # 恢复持续时间达到配置上限时触发超时；该检查先于数据等待分支。
            self.status.update(pid_recovery=False,  # 清除日志中的 PID 恢复请求标志。
                               wrench_recovery_seconds=now - self.wrench_recovery_started_at)  # 记录此次力/力矩恢复累计持续的秒数。
            return self._hold('wrench_recovery_timeout', latch=True)  # 锁存恢复超时故障并返回保持指令。
        if self.previous_time is None:  # 首次调用尚无上一控制时刻时，初始化时间基准。
            self.previous_time = now - config.control_period  # 将上一时刻设置为一个名义控制周期之前。
            self.last_progress = now  # 初始化最近一次轨迹推进的时刻。
        elapsed = now - self.previous_time  # 计算两次调用之间的实际时间间隔，单位为秒。
        self.previous_time = now  # 保存当前时刻，作为下次调用计算间隔的基准。
        dt = min(max(elapsed, 0.001), 0.01)  # 将优化使用的时间步长限制在 1～10 毫秒，避免异常间隔放大位移。
        sensor_times = np.asarray(sensor_times, dtype=float)  # 将传感器时间戳转换为浮点数组，供新鲜度检查使用。
        if elapsed > config.sensor_timeout or self.control_gap_wait_started_at is not None:  # 调用间隔超限或此前仍在等待数据时，进入控制间断恢复分支。
            if self.control_gap_wait_started_at is None:  # 仅在首次发现此次控制间断时初始化等待状态。
                self.control_gap_wait_started_at = now  # 记录等待开始时刻，要求后续输入全部晚于该时刻。
                self.control_gap_recovering = True  # 标记仍处于控制间断恢复状态，用于限制姿态更新。
                self.stable_frames = 0  # 清零稳定帧计数，要求恢复后重新积累稳定证据。
                self.friction_recovery_started_at = None  # 清除摩擦恢复计时，避免沿用间断前的恢复时长。
                self.previous_force_world = None  # 清除上一力向量，防止跨数据间断计算不连续的力差。
                self.force_change[:] = 0.0  # 将本轮力变化向量归零。
                self.status = dict(control_gap_seconds=elapsed)  # 重新建立状态字典并记录触发间断的时间间隔。
            self.last_progress += elapsed  # 将等待时间补偿到推进时刻，使等待不计入无进展超时。
            self.previous_velocity[:] = 0.0  # 清零上一指令速度，避免恢复时沿用旧速度。
            fresh_inputs = (  # 组合判断此次输入是否满足严格的数据恢复条件。
                sensor_times.shape == (7,) and np.all(np.isfinite(sensor_times))  # 要求恰好七个传感器时间戳，且全部为有限数值。
                and np.all(sensor_times > self.control_gap_wait_started_at)  # 要求所有时间戳均晚于本次等待开始时刻。
                and np.all((now - sensor_times >= 0.0)  # 要求时间戳不在当前时刻之后。
                           & (now - sensor_times <= config.sensor_timeout)))  # 同时要求全部数据年龄不超过传感器超时阈值。
            if elapsed > config.sensor_timeout or not fresh_inputs:  # 调用间隔仍过大或输入尚未完全更新时，继续等待。
                self.status.update(  # 更新等待阶段的诊断信息。
                    waiting_seconds=now - self.control_gap_wait_started_at,  # 记录从进入等待到当前的持续时间。
                    stable_frames=0, solver_skipped=True, solver_success=False,  # 标明未积累稳定帧、跳过求解且没有有效求解结果。
                    solver_ms=0.0, controller_ms=1000.0 * (time.perf_counter() - started_at))  # 记录零求解耗时及本次控制调用截至此处的耗时。
                return self._hold('control_gap', mode='waiting_data')  # 返回等待数据模式的保持指令；该路径不锁存故障。
            self.control_gap_wait_started_at = None  # 输入恢复正常后清除等待起始时刻。
            elapsed = 0.0  # 本轮用于恢复计时补偿的实际间隔归零。
            dt = config.control_period  # 以名义控制周期恢复优化计算。
        if (sensor_times.size == 0 or not np.all(np.isfinite(sensor_times))  # 普通运行路径检查时间戳是否为空或包含非有限数值。
                or now - np.min(sensor_times) > config.sensor_timeout):  # 同时检查最旧的数据是否已超过允许年龄。
            self.stable_frames = 0  # 传感器无效或过期时清零稳定帧。
            return self._hold('stale_sensor', latch=True)  # 锁存传感器过期故障并保持当前指令。
        if not contact_active:  # 检查外部接触激活标志。
            self.stable_frames = 0  # 接触未激活时清零稳定帧计数。
            return self._hold('contact_inactive', latch=True)  # 锁存接触未激活故障；此处不同于后续低法向力的 PID 恢复。
        values = (position, quat, force, moment, target_position, target_quat, pivot)  # 收集必须为有限值的位姿、力矩、目标和参考点输入。
        if not all(np.all(np.isfinite(value)) for value in values):  # 逐项检查输入是否含有 NaN 或无穷值。
            return self._hold('nonfinite_input', latch=True)  # 输入存在非有限值时锁存故障并保持。
        position = np.asarray(position, dtype=float)  # 将当前末端位置转换为浮点数组，单位为米。
        target_position = np.asarray(target_position, dtype=float)  # 转换目标位置；当前优化并不将位置跟踪误差纳入目标函数。
        quat = np.asarray(quat, dtype=float)  # 将当前四元数转换为浮点数组，顺序为 SciPy 使用的 x、y、z、w。
        target_quat = np.asarray(target_quat, dtype=float)  # 将目标四元数转换为相同格式。
        if np.linalg.norm(quat) < 1e-9 or np.linalg.norm(target_quat) < 1e-9:  # 检查当前和目标四元数是否接近零向量。
            return self._hold('invalid_quaternion', latch=True)  # 无效四元数无法定义旋转，锁存故障并保持。

        rotation = Rotation.from_quat(quat)  # 由当前测量四元数构造旋转对象。
        target_rotation = Rotation.from_quat(target_quat)  # 由当前轨迹目标四元数构造目标旋转。
        angle_error = (rotation * target_rotation.inv()).magnitude()  # 计算当前真实姿态与目标姿态间的旋转角误差，单位为弧度。
        if not self.control_gap_recovering:  # 非控制间断恢复期间，允许先更新内部姿态指令。
            self.command_rotation = (target_rotation if self.friction_hold_rotation is None  # 没有摩擦冻结姿态时，直接采用当前轨迹目标姿态。
                                     else self.friction_hold_rotation)  # 存在摩擦冻结姿态时，继续采用该冻结姿态。
        sensor_rotation = rotation * self.force_reference_rotation.inv()  # 计算当前旋转相对于初始力参考旋转的坐标变换。
        force_world = sensor_rotation.apply(np.asarray(force, dtype=float) + self.force_bias) - self.force_bias  # 先加回力偏置，再旋转到参考坐标表示，最后减去参考系力偏置。
        moment_world = sensor_rotation.apply(np.asarray(moment, dtype=float))  # 旋转力矩向量；这里没有进行力矩参考点平移补偿。
        normal_force = float(np.dot(self.normal, force_world))  # 沿固定接触法向投影力向量，得到带符号法向力。
        tangent_vector = self.tangent_projector @ force_world  # 将力投影到法向正交平面，得到切向力向量。
        tangent_force = float(np.linalg.norm(tangent_vector))  # 求切向力向量的模长，单位为牛顿。
        force_norm = float(np.linalg.norm(force_world))  # 计算变换后的合力模长。
        moment_norm = float(np.linalg.norm(moment_world))  # 计算变换后的力矩模长，沿用输入力矩单位。
        self.status = dict(normal_force=normal_force, tangent_force=tangent_force,  # 建立本轮诊断字典，保存法向力与切向力。
                           force_norm=force_norm, moment_norm=moment_norm,  # 保存合力和力矩模长。
                           mu_safe=self.mu_safe, angle_error=float(angle_error),  # 保存当前安全摩擦系数及弧度制姿态误差。
                           angle_error_deg=float(np.rad2deg(angle_error)))  # 同时保存角度制姿态误差，便于阅读日志。
        wrench_limit = normal_force <= 0.1 or force_norm > 11.0 or moment_norm > 50.0  # 法向力过低、合力超过 11 或力矩超过 50 时触发极限恢复条件。
        if wrench_limit or self.wrench_recovery_started_at is not None:  # 达到极限或已在 PID 恢复过程中时，处理力/力矩恢复状态。
            if self.wrench_recovery_started_at is None:  # 首次进入 PID 恢复时才初始化计时和冻结姿态。
                self.wrench_recovery_started_at = now  # 记录恢复开始时刻，后续重复触发不会重置该时刻。
                self.wrench_recovery_rotation = rotation  # 以当前测量姿态作为恢复期间固定的姿态。
                self.stable_frames = 0  # 清零稳定帧，要求退出恢复后重新确认稳定性。
                self.friction_recovery_started_at = None  # 清除摩擦恢复计时，避免与 PID 恢复混用。
            target_range_adjust = (  # 计算 PID 恢复结束所要求的力/力矩组合范围。
                (0.5 < moment_norm < 30.0 and 2.1 < force_norm < 2.5)  # 第一种允许组合：力矩在 0.5～30 且合力在 2.1～2.5。
                or (18.0 < moment_norm < 22.0 and 0.7 < force_norm < 2.0))  # 第二种允许组合：力矩在 18～22 且合力在 0.7～2.0。
            self.command_rotation = self.wrench_recovery_rotation  # 在 PID 恢复分支中使用冻结姿态。
            if wrench_limit or not target_range_adjust:  # 仍触及极限或尚未达到恢复目标范围时，继续请求外部 PID 调整。
                self.status.update(pid_recovery=True,  # 设置 PID 恢复请求标志，交由调用节点执行其力控制逻辑。
                                   wrench_recovery_seconds=now - self.wrench_recovery_started_at,  # 记录 PID 恢复持续时间。
                                   solver_skipped=True, solver_success=False, solver_ms=0.0)  # 注明本轮不执行位置优化，因此没有求解结果和求解耗时。
                return self._hold('contact_lost_or_wrench_limit', mode='recovery')  # 返回恢复模式的保持输出；本类自身不在这里计算 PID 位置增量。
            self.wrench_recovery_started_at = None  # 满足退出条件后清除 PID 恢复计时。
            self.wrench_recovery_rotation = None  # 清除 PID 恢复冻结姿态。
            self.last_progress = now  # 重新开始无进展计时，避免将已完成的恢复时间计入失败。
            self.previous_force_world = None  # 清除力差计算的上一帧基准。
            self.previous_velocity[:] = 0.0  # 清零上一指令速度，重新开始位置增量优化。
            force_control = False  # 仅在本次调用内清除外部力控制参数的局部副本。

        pose_error = self.command_position - position  # 计算上一位置指令与当前位置的跟踪偏差。
        if np.linalg.norm(pose_error) > 0.001:  # 位置跟踪偏差超过 1 毫米时触发故障。
            return self._hold('robot_tracking_error', latch=True)  # 锁存机械臂位置跟踪误差并保持。

        fresh_objects = (self.sensor_stamps is None or  # 首次获取点云，或双侧点云时间戳均已更新时，视为新物体观测。
                          all(stamp > previous for stamp, previous in  # 同时逐一比较本次与上次的两侧点云时间戳。
                              zip(sensor_times[:2], self.sensor_stamps)))  # 这里只使用七个传感器时间戳中的前两个。
        try:  # 尝试估计当前双侧点云相对于参考点云的刚体变换。
            objects = [self._fit_cloud(reference, np.asarray(current))  # 逐侧执行点云拟合，并将当前点云转换为数组。
                       for reference, current in zip(self.reference_clouds, clouds)]  # 通过对应的参考点云和当前点云构造拟合结果列表。
        except (ValueError, np.linalg.LinAlgError):  # 点云格式错误或线性代数计算失败时进入异常处理。
            return self._hold('invalid_object_fit', latch=True)  # 锁存点云拟合输入或计算失败故障。
        if len(objects) != 2 or any(result[2] > config.fit_residual for result in objects):  # 要求正好两侧拟合结果，且每侧残差均不超过配置阈值。
            return self._hold('object_fit_residual', latch=True)  # 点云数量或拟合质量不合要求时锁存故障并保持。
        stable = True  # 初始化本轮点云稳定判断为真，后续新观测可将其改为假。
        object_jump = False  # 初始化本轮物体姿态突变标志为假。
        object_motion = 0.0  # 初始化本轮归一化物体运动指标。
        if fresh_objects:  # 仅在双侧点云均有新时间戳时更新运动状态及稳定计数。
            object_dt = (min(max(now - self.object_stamp, 0.001), config.sensor_timeout)  # 已有上一观测时，将观测间隔限制在 1 毫秒至传感器超时之间。
                         if self.object_stamp is not None else dt)  # 首次物体观测使用本轮控制时间步长。
            if self.previous_objects is not None:  # 存在上一组拟合结果时，才能计算相邻观测的位姿变化。
                for current, previous in zip(objects, self.previous_objects):  # 逐侧比较当前和上一组点云拟合结果。
                    distance = np.linalg.norm(current[0] - previous[0])  # 计算该侧点云质心变化距离，单位为米。
                    angle = (current[1] * previous[1].inv()).magnitude()  # 计算该侧拟合旋转的相邻角度变化，单位为弧度。
                    object_motion += (  # 累计两侧归一化物体运动强度。
                        (distance / config.object_jump_distance) ** 2  # 将质心距离除以突变距离阈值并平方。
                        + (angle / config.object_jump_angle) ** 2)  # 将旋转角除以突变角度阈值并平方后加入指标。
                    object_jump |= (distance > config.object_jump_distance + config.object_linear_speed * object_dt  # 质心变化超过基础距离阈值加速度容许量时，标记突变。
                                    or angle > config.object_jump_angle + config.object_angular_speed * object_dt)  # 旋转变化超过基础角度阈值加角速度容许量时，同样标记突变。
                    stable &= (distance <= 0.25 * config.object_jump_distance + config.object_linear_speed * object_dt  # 稳定判据使用四分之一基础距离阈值加线速度容许量。
                               and angle <= 0.25 * config.object_jump_angle + config.object_angular_speed * object_dt)  # 同时要求旋转变化满足四分之一基础角度阈值加角速度容许量。
            self.previous_objects = objects  # 保存当前双侧拟合结果，作为下一次新观测的比较基准。
            self.object_stamp = now  # 保存处理该次物体观测的控制时刻。
            self.sensor_stamps = tuple(sensor_times[:2])  # 保存双侧点云时间戳，用于检测下一次新观测。
            if not stable and not self.object_unstable:  # 首次发现非稳定观测时设置物体不稳定状态。
                self.object_unstable = True  # 保持该状态，直到连续稳定帧数量满足恢复要求。
            self.stable_frames = self.stable_frames + 1 if stable else 0  # 稳定则计数加一，不稳定则归零；重复旧点云不会增加计数。
            if self.stable_frames >= config.recovery_frames:  # 连续稳定观测达到恢复帧数时，允许清除不稳定状态。
                self.object_unstable = False  # 清除已恢复的物体不稳定标记。
        if fresh_objects:  # 仅在新物体观测时更新力变化相关指标。
            self.object_motion = float(object_motion)  # 保存本轮双侧归一化物体运动强度。
            if self.previous_force_world is None:  # 没有上一力向量时，无法计算真实相邻力差。
                self.force_change[:] = 0.0  # 将首次力变化设为零向量。
            else:  # 存在上一力向量时执行差分。
                self.force_change = force_world - self.previous_force_world  # 计算当前参考系力向量减去上一力向量，保留变化方向。
            self.object_wrench_change = float(np.linalg.norm(self.force_change))  # 记录力变化模长；该指标不包含力矩变化。
            self.previous_force_world = force_world.copy()  # 复制当前力向量，供下一次新观测计算差分。

        self.status.update(stable_frames=self.stable_frames,  # 更新连续稳定观测帧数。
                           recovery_frames=config.recovery_frames,  # 记录恢复所要求的稳定帧数。
                           no_progress_seconds=now - self.last_progress)  # 记录距最近轨迹推进的时间，包含当前已有补偿。

        normal_ok = config.normal_min <= normal_force <= config.normal_max  # 法向力是否在 2～2.5 N 仅作为目标范围诊断，不直接决定跟踪许可。
        measured_wrench_range_ok = (0.5 < force_norm < 10.0 and 0.05 < moment_norm < 40.0)  # 计算本类默认的合力与力矩允许范围。
        wrench_range_ok = (measured_wrench_range_ok if wrench_in_target is None  # 调用方未给范围判断时，使用本类的默认范围结果。
                           else bool(wrench_in_target))  # 调用方提供判断时，以其布尔结果为准。
        self.mu_safe = min(self.mu_safe, self.mu_max_estimate)  # 将安全摩擦系数限制为不超过当前最大摩擦系数估计值。
        friction_ok = tangent_force + config.force_reserve <= self.mu_safe * normal_force  # 将切向力加预留量与安全摩擦承载力比较，判断是否满足摩擦约束。
        stable_ok = self.stable_frames >= config.recovery_frames  # 连续稳定帧达到阈值时，认为稳定性恢复完成。
        normal_target = self.normal_force_reference  # 使用 N3 开始前稳定接触窗口得到的法向参考力。
        recovery_eligible = (not friction_ok and stable_ok and not object_jump  # 只有摩擦约束不满足且物体已稳定、没有突变时，才可能进入摩擦恢复计时。
                             and wrench_range_ok and not force_control)  # 还要求 wrench 在允许范围内，且外部 PID 力控制没有运行。
        if recovery_eligible:  # 满足摩擦恢复资格时处理恢复计时。
            if self.friction_recovery_started_at is None:  # 首次进入本次摩擦恢复时初始化计时。
                self.friction_recovery_started_at = now  # 记录摩擦恢复开始时刻。
            recovery_seconds = now - self.friction_recovery_started_at  # 计算本次摩擦恢复已持续的秒数。
            self.last_progress += max(  # 将本次调用中属于允许恢复时段的时间补偿到最近推进时刻。
                0.0, min(now, self.friction_recovery_started_at + config.friction_recovery_timeout)  # 最多补偿摩擦恢复开始后的配置恢复时限，当前为 10 秒。
                - max(now - elapsed, self.friction_recovery_started_at))  # 仅补偿本轮实际覆盖的恢复时间区间，避免重复补偿。
        else:  # 不符合摩擦恢复资格时取消本次恢复计时。
            self.friction_recovery_started_at = None  # 清除摩擦恢复开始时刻。
            recovery_seconds = 0.0  # 将本轮显示的摩擦恢复持续时间置零。
        self._update_friction_estimate(  # 调用独立摩擦系数更新函数。
            tangent_force, normal_force, dt, stable_ok, object_jump,  # 传入当前切向力、法向力、时间步长和稳定/突变状态。
            wrench_range_ok, friction_ok, fresh_objects, allow_update=not force_control,  # 传入范围、摩擦及新观测判断；外部力控制期间禁止更新估计。
            allow_increase=recovery_eligible and recovery_seconds >= config.friction_recovery_timeout)  # 只有满足恢复资格且先尝试位置恢复达到时限后，才允许增大安全摩擦系数。
        friction_limit = self.mu_safe * normal_force  # 用更新后的安全摩擦系数计算当前摩擦承载力。
        friction_ok = tangent_force + config.force_reserve <= friction_limit  # 重新检查实际测量力是否满足带预留量的摩擦约束。
        friction_exceeded = not friction_ok  # 记录是否超出当前安全摩擦边界。
        if friction_exceeded:  # 超出摩擦边界时冻结姿态。
            if self.friction_hold_rotation is None:  # 仅在首次超界时建立冻结姿态。
                self.friction_hold_rotation = rotation  # 冻结值采用首次超界时测量的真实姿态。
        else:  # 回到摩擦边界内时解除摩擦冻结状态。
            self.friction_hold_rotation = None  # 清除摩擦冻结姿态。
            self.friction_recovery_started_at = None  # 清除已结束的摩擦恢复计时。
        tracking = (not force_control and not object_jump and wrench_range_ok  # 跟踪许可要求未进行外部力控制、无物体突变且 wrench 正常。
                    and friction_ok and stable_ok)  # 同时要求实际摩擦约束满足，且已积累足够稳定帧。
        if tracking:  # 所有跟踪条件满足时，结束控制间断后的恢复状态。
            self.control_gap_recovering = False  # 清除控制间断恢复标志，允许正常采用目标姿态。
        if self.friction_hold_rotation is not None:  # 有摩擦冻结姿态时优先采用该姿态。
            self.command_rotation = self.friction_hold_rotation  # 将姿态指令设置为摩擦恢复的冻结姿态。
        elif not self.control_gap_recovering:  # 无摩擦冻结且不再处于控制间断恢复时，可以更新目标姿态。
            self.command_rotation = target_rotation  # 直接将当前轨迹目标旋转赋给姿态指令，不做独立角速度积分。
        self.status.update(friction_recovery_seconds=recovery_seconds,  # 记录摩擦恢复累计秒数。
                           no_progress_seconds=now - self.last_progress)  # 记录经过恢复补偿后的无进展持续时间。
        if now - self.last_progress > config.progress_timeout:  # 无进展持续时间超过配置上限时，判定无法继续推进。
            return self._hold('no_progress_timeout', latch=True)  # 锁存无进展超时故障并保持。
        friction_adapting = (friction_exceeded and recovery_eligible  # 摩擦超界且其他恢复条件满足时，检查是否进入系数增大阶段。
                             and recovery_seconds >= config.friction_recovery_timeout)  # 只有先尝试位置恢复达到时限后，才标记为摩擦系数调整阶段。
        if previous_mode == 'tracking' and not stable_ok and not object_jump:  # 上轮为跟踪、本轮稳定帧不足且没有突变标志时，对安全摩擦系数折减。
            self.mu_safe *= 0.99  # 仅在跟踪转为 object_pose_unstable 恢复时小幅降低安全摩擦系数。
            friction_limit = self.mu_safe * normal_force  # 用折减后的系数重算摩擦承载力。
            friction_ok = tangent_force + config.force_reserve <= friction_limit  # 重算摩擦约束判断；本轮更早形成的 tracking 值不在这里重新计算。
            friction_exceeded = not friction_ok  # 更新摩擦超界标志，供后续状态记录与原因选择使用。
        if object_jump or not stable_ok:  # 存在物体突变或稳定帧不足时，本轮跳过位置优化。
            optimized_position = position.copy()  # 以当前测量位置作为位置指令，不增加优化位移。
            optimized_delta = np.zeros(3, dtype=float)  # 将位置增量置为三维零向量。
            objective_before = self._objective(  # 为诊断仍计算零增量下的目标函数值。
                optimized_delta, force_world, dt, self.object_motion,  # 传入当前力、控制周期和已有物体运动强度。
                self.force_change, self.mu_safe)  # 使用当前力变化及安全摩擦系数评价目标。
            objective_after = objective_before  # 未执行优化，优化后目标值与优化前相同。
            self.optimization_status = dict(  # 建立跳过求解时的优化状态记录。
                solver_success=False, solver_skipped=True, solver_ms=0.0,  # 标记没有成功求解、本轮已跳过求解且求解耗时为零。
                solver_iterations=0, solver_budget_exceeded=False,  # 记录零迭代次数，并注明没有求解预算超限。
                solver_message='object_not_stable')  # 使用物体尚未稳定作为跳过求解的说明。
        else:  # 物体稳定时执行位置优化。
            optimized_position, optimized_delta, objective_before, objective_after = (  # 接收优化位置、位移增量以及优化前后的目标函数值。
                self._optimize_position(  # 调用受约束的当前位置增量优化器。
                    position, force_world, dt, self.object_motion,  # 输入当前位置、当前力、时间步长及物体运动强度。
                    self.force_change, self.mu_safe, enforce_cone=tracking))  # 传入力差和安全摩擦系数；仅跟踪模式启用预测摩擦锥硬约束。
            if optimized_position is None:  # 优化结果无效时处理求解失败。
                self.command_position = position.copy()  # 将位置指令重置为当前测量位置，避免保留过时的位置目标。
                self.status.update(self.optimization_status)  # 合并本次求解诊断信息。
                self.status.update(controller_ms=1000.0 * (time.perf_counter() - started_at))  # 记录本次调用截至失败处理处的控制耗时。
                return self._hold('optimization_failed')  # 返回非锁存的优化失败保持输出，后续调用可再次尝试。
        velocity = np.zeros(6, dtype=float)  # 初始化六维指令速度向量，前三项平移、后三项旋转。
        velocity[:3] = optimized_position - position  # 用优化后的绝对位置减去当前测量位置得到平移指令增量。
        velocity[:3] /= max(dt, 1.0e-3)  # 除以受下限保护的时间步长，得到本轮平移指令速度。
        velocity[3:] = 0.0  # 将内部角速度字段置零；这不代表机械臂实际没有转动。
        self.command_position = optimized_position  # 保存本轮优化后的位置指令。
        self.previous_velocity = velocity  # 保存本轮指令速度，用于下一轮平滑目标和加速度约束。
        settled = np.linalg.norm(velocity[:3]) < 0.0001  # 平移指令速度模长低于 0.1 毫米/秒时，认为位置调整趋于静止。
        advance = (tracking and angle_error <= config.angle_tolerance  # 推进轨迹要求处于跟踪模式且真实姿态误差不超过容差。
                   and (settled or not final_point))  # 末端路径点还要求平移调整趋于静止，普通路径点无需该条件。
        if advance:  # 允许推进时刷新推进时间基准。
            self.last_progress = now  # 保存本次成功推进时刻，重置无进展计时。
        reason = ('object_pose_jump' if object_jump else  # 恢复原因首先优先记录物体姿态突变。
                  'object_pose_unstable' if not stable_ok else  # 其次记录稳定帧不足导致的物体姿态不稳定。
                  'force_control' if force_control else  # 再次记录外部 PID 力控制状态。
                  'friction_adapting' if friction_adapting else  # 随后记录摩擦系数调整阶段。
                  'friction_recovery' if friction_exceeded else '')  # 最后记录摩擦恢复；以上条件均不满足时原因为空字符串。
        self.status.update(mode='tracking' if tracking else 'recovery', reason=reason,  # 根据此前计算的跟踪许可记录模式，并保存本轮原因。
                           mu_safe=self.mu_safe,  # 记录最终安全摩擦系数。
                           stable_frames=self.stable_frames,  # 记录当前连续稳定帧数。
                           recovery_frames=config.recovery_frames,  # 记录恢复所需的稳定帧阈值。
                           normal_force_reference=self.normal_force_reference,  # 记录 N3 使用的法向参考力。
                           tangent_force_reference=self.tangent_force_reference,  # 记录 N3 使用的切向参考力。
                           normal_in_target=normal_ok,  # 记录法向力是否处于软目标范围。
                           normal_error=normal_force - normal_target,  # 记录法向力相对区间中点目标的有符号误差。
                           wrench_in_target=wrench_range_ok,  # 记录实际采用的 wrench 范围判断。
                           measured_wrench_in_target=measured_wrench_range_ok,  # 同时保留本类独立计算的范围判断，便于与外部判断对照。
                           stable_ok=bool(stable_ok),  # 记录稳定性是否满足要求。
                           object_jump=bool(object_jump),  # 记录本轮物体姿态突变标志。
                           hard_constraints_ok=bool(  # 汇总测量状态的硬约束满足情况。
                               wrench_range_ok and stable_ok and not object_jump and friction_ok),  # 该汇总要求 wrench 正常、稳定、没有突变且摩擦约束满足。
                           friction_limit=friction_limit,  # 记录安全摩擦承载力，即安全摩擦系数乘以法向力。
                           friction_exceeded=friction_exceeded,  # 记录是否超出带预留量的安全摩擦边界。
                           friction_adapting=friction_adapting,  # 记录是否处于摩擦系数调整阶段。
                           tracking_allowed=tracking,  # 记录此前计算的跟踪许可；跟踪许可不等同于轨迹已推进。
                           advance=bool(advance), settled=bool(settled),  # 记录实际推进标志和平移调整静止标志。
                           linear_speed=float(np.linalg.norm(velocity[:3])),  # 记录优化位移除以时间步长得到的平移指令速度模长。
                           angular_speed=0.0,  # 角速度日志固定为零，并非测量得到的实际角速度。
                           target_angular_rate=float(angle_error / max(dt, 1.0e-3)),  # 将当前姿态误差除以时间步长作为目标角速度指标，不是实际角速度。
                           force_ratio_mean=float(np.mean(self.force_ratios)) if self.force_ratios else None,  # 记录最近至多五个有效切向力/法向力比值的均值，无样本时为 None。
                           mu_max_estimate=self.mu_max_estimate,  # 记录当前保留的最大摩擦系数估计。
                           objective_before=float(objective_before),  # 记录执行当前位移增量前的目标函数值。
                           objective_after=float(objective_after),  # 记录执行优化位移增量后的预测目标函数值。
                           optimized_delta_position=optimized_delta.copy(),  # 复制并记录本轮优化位置增量，单位为米。
                           object_motion=float(self.object_motion),  # 记录最近一次新物体观测的归一化运动强度。
                           object_wrench_change=float(self.object_wrench_change),  # 记录最近一次新物体观测对应的力变化模长。
                           optimized_next_position=optimized_position.copy(),  # 复制并记录下一位置指令，单位为米。
                           command_velocity=velocity.copy())  # 复制并记录六维内部指令速度。
        self.status.update(self.optimization_status)  # 将求解器诊断信息合并到控制状态字典。
        controller_ms = 1000.0 * (time.perf_counter() - started_at)  # 计算完整控制调用截至此处的耗时，单位为毫秒。
        self.status.update(controller_ms=controller_ms,  # 记录本轮控制耗时。
                           controller_budget_exceeded=controller_ms > 1000.0 * config.control_period)  # 记录是否超过名义控制周期预算；该状态字段本身不会中断执行。
        return self.command_position.copy(), self.command_rotation.as_quat(), advance  # 返回位置指令副本、四元数姿态指令以及轨迹推进许可。
