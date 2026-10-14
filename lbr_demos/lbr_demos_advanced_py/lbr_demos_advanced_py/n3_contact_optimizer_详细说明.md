# n3_contact_optimizer.py 详细中文说明

本文对应当前源码。此次仅补充逐行中文注释和本说明，没有调整控制逻辑、参数默认值、接口、目标函数或约束。历史讨论中的参数不一定等于当前文件中的数值，以下以当前实现为准。

## 1. 文件的作用与控制边界

该文件实现 N3 偏转阶段的有状态单步控制器。每次调用接收当前末端位姿、力/力矩、触觉点云和轨迹目标，输出下一位置指令、四元数姿态指令，以及是否允许原始轨迹推进的布尔值。

位置通过一个三维受约束优化问题计算。正常跟踪时，姿态指令直接采用当前目标四元数，不通过独立角速度积分生成；摩擦恢复等特定状态会冻结姿态。

本文件自身不创建 ROS 线程、订阅或定时器，也不直接发布指令。调用频率、消息收集、轨迹索引和最终发布由规划节点管理。返回了指令不代表机械臂已经到达；advance=True 才表示控制器允许调用方推进原始轨迹。

当前规划节点的 _publish_n3_optimized 调用本控制器；当状态要求 pid_recovery=True 时，节点转由 _apply_n3_wrench_recovery_pid 计算恢复位置。

## 2. 类和方法一览

| 类/方法 | 主要职责 |
|---|---|
| N3ContactConfig | 保存默认参数，允许调用者覆盖 |
| N3ContactOptimizer.__init__ | 校验初始输入，保存法向、刚度和参考点云，初始化状态 |
| _prepare_stiffness | 刚度回退、对称化和特征值限幅 |
| _skew | 构造叉乘矩阵，当前内部控制路径没有调用 |
| _fit_cloud | 对对应触觉点云作刚体拟合，返回质心、旋转和残差 |
| start_segment | 新轨迹段开始时重置部分计时和历史 |
| _hold | 返回保存的位姿并禁止推进，可选择锁存故障 |
| _update_friction_estimate | 更新观测力比历史及最大值，按条件增加安全摩擦系数 |
| _objective_and_gradient | 计算米制位置增量对应的完整目标值及解析梯度 |
| _objective | 仅返回上述目标值 |
| _optimize_position | 对归一化位置变量调用 SLSQP，校验可行性和耗时 |
| step | 执行输入检查、恢复判断、稳定判断、摩擦更新和优化 |

## 3. 输入、输出、单位及坐标系

### 3.1 主要输入

| 输入 | 含义及要求 |
|---|---|
| position | 当前实测末端位置，三维，单位 m |
| quat | 当前实测四元数，顺序为 [x,y,z,w] |
| force | 调用方提供的三维接触力，单位 N |
| moment | 三维力矩；阈值必须与输入单位一致，本类不换算力矩单位 |
| target_position | 当前原始轨迹位置；只作有限性检查和数组转换，不进入位置跟踪目标 |
| target_quat | 当前原始轨迹姿态 |
| pivot | 传入参考点；当前仅参与有限性检查，不用于位置优化 |
| clouds | 左右两侧对应触觉标记点云，每侧为至少三个非退化点组成的 N×3 数组，单位 m |
| sensor_times | 传感器接收时间戳，与 now 使用相同单调时间基准 |
| contact_active | 外部接触激活判断 |
| force_control | 外部是否正在进行力控制 |
| wrench_in_target | 外部允许范围判断；为 None 时采用本类默认判断 |
| final_point | 是否为当前轨迹段最后一点 |
| now | 可选当前时刻；默认使用 time.monotonic() |

当前节点按 P_l、P_r、Fr_l、Fr_r、F_l、F_r、pose 顺序传入七个时间戳。前两个用于判断双侧点云是否都有新观测。节点先将触觉点坐标变换到共同表示并从 mm 转为 m。

输出为 command_position 的副本、command_rotation 对应四元数、advance。状态日志另存于 status 字典。

### 3.2 力的坐标变换

记当前测量旋转为 \(R\)，创建控制器时保存的旋转为 \(R_0\)，力偏置为 \(b\)。代码计算：

\[
R_s=R R_0^{-1},
\qquad f=R_s(f_{\mathrm{input}}+b)-b,
\qquad m=R_s m_{\mathrm{input}}.
\]

这里先加回偏置，再旋转，再减去参考偏置，不同于直接计算 \(R_s f_{\mathrm{input}}\)。

需要注意：

- force_reference_rotation 是创建本控制器时的姿态，未必是程序最初启动时的姿态。
- 只有参考时刻夹爪坐标轴与世界坐标轴对齐、输入力和偏置定义也相容时，才能按对应世界坐标解释结果。
- 程序最初坐标轴对齐，不自动保证之后创建 N3 控制器时仍然对齐。
- 法向、刚度、优化位移和变换后的力必须采用相容的坐标表示。
- 力矩仅作旋转，没有加入改变参考点时所需的力臂叉乘补偿。

变量名 force_world 表达代码意图，不能代替传感器坐标定义核对。本次只解释现有公式，没有改变它。

### 3.3 法向与切向

初始化时归一化输入法向：

\[
n=\frac{n_{\mathrm{input}}}{\|n_{\mathrm{input}}\|},
\qquad P_t=I-nn^\top.
\]

随后计算：

\[
f_n=n^\top f,\qquad \boldsymbol f_t=P_t f,\qquad f_t=\|\boldsymbol f_t\|.
\]

法向力有正负号；切向力是完整切平面投影的模长，不是单独的 \(f_x\) 或 \(f_y\)。法向初始化后保持不变，本类不在线重新辨识法向。

## 4. 当前默认参数

下表列出文件默认值。当前规划节点创建配置时会用 self.velocity 覆盖 linear_speed，用 self.rotation_w 覆盖 angular_speed，因此默认值不一定等于实际运行值。

| 类别 | 字段 | 当前默认值 |
|---|---|---|
| 法向目标 | normal_min / normal_max | 2.0 / 2.5 N，中点 2.25 N |
| 摩擦裕量 | force_reserve | 0.02 N |
| 摩擦先验 | mu_prior / mu_floor | 0.18 / 0.05 |
| 初始安全比例 | mu_safe_factor | 0.7 |
| 摩擦增长倍率 | mu_safe_growth | 1.1 |
| 摩擦位置恢复时间 | friction_recovery_timeout | 10 s |
| 提前惩罚比例 | friction_penalty_start | 0.8 |
| 平移速度上限 | linear_speed | 0.0045 m/s |
| 逐轴平移加速度限制 | linear_acceleration | 0.01 m/s² |
| 物体位移阈值速度项 | object_linear_speed | 0.002 m/s |
| 物体角度阈值速度项 | object_angular_speed | 3°/s |
| 位移突变基础阈值 | object_jump_distance | 0.00008 m，即 0.08 mm |
| 角度突变基础阈值 | object_jump_angle | 0.12° |
| 点云拟合残差上限 | fit_residual | 0.0001 m，即 0.1 mm |
| 目标姿态容差 | angle_tolerance | 0.08° |
| 单周期位移限额 | correction_limit | 0.002 m，即 2 mm |
| 数据/调用间隔超时 | sensor_timeout | 0.2 s |
| 无进展超时 | progress_timeout | 10 s |
| 极限 wrench PID 恢复超时 | wrench_recovery_timeout | 10 s |
| 连续稳定观测要求 | recovery_frames | 10 帧 |
| 刚度特征值范围 | stiffness_floor / stiffness_ceiling | 100～100000 N/m |
| 求解设置 | optimization_iterations / optimization_tolerance | 30 次 / \(10^{-9}\) |
| 求解预算 | optimization_budget_ms | 6 ms |
| 名义控制周期 | control_period | 0.01 s，即 100 Hz |

目标权重：法向平方误差 4、低法向力线性惩罚 10、切向力平方 8、摩擦边界惩罚 100、速度平滑 0.2、物体稳定代理项 50、力变化连续项 0.5。

当前保留但没有被本文件控制计算读取的配置字段包括：mu_factor、mu_increase_rate、mu_max、angular_speed、angular_acceleration、normal_recovery_gain、normal_recovery_speed、force_control_gain、position_tolerance。

例如，安全摩擦系数的实际增长上限是 mu_max_estimate，并不是配置 mu_max=1.5；位置跟踪异常阈值直接写为 0.001 m，没有读取 position_tolerance。

此外，failures 仅初始化，_skew 当前内部未调用，_update_friction_estimate 的 object_dt 参数没有参与计算。

## 5. 刚度模型与位置变量

### 5.1 刚度预处理

输入必须是有限的 3×3 矩阵，否则回退为 \(K=100I\)。有效输入先对称化，再限幅特征值：

\[
K_s=\frac{K_{\mathrm{in}}+K_{\mathrm{in}}^\top}{2},
\quad K_s=Q\Lambda Q^\top,
\quad K=Q\operatorname{diag}(\operatorname{clip}(\lambda_i,100,100000))Q^\top.
\]

所以输入刚度不会原样使用：小特征值和负特征值提高到下限，大特征值压到上限，非对角耦合通常仍然存在。模型在控制器创建后保持固定。

### 5.2 下一位置与预测力

优化变量是三维位置增量 \(\delta p\)：

\[
p_{\mathrm{next}}=p_{\mathrm{measured}}+\delta p,
\qquad \delta f=-K\delta p,
\qquad \hat f=f+\delta f.
\]

预测法向、切向为：

\[
\hat f_n=n^\top\hat f,\qquad
\hat{\boldsymbol f}_t=P_t\hat f,\qquad
\hat f_t=\|\hat{\boldsymbol f}_t\|.
\]

该模型只描述平移引起的力变化，没有显式预测目标旋转引起的力变化，也没有力矩预测模型，属于小增量近似。

没有原始位置跟踪误差项，因此位置围绕当前实测位置连续调整，累计位移可能增加。correction_limit 只限制单周期增量，不限制相对 N3 初始位置的累计漂移。

## 6. 完整目标函数

定义：

\[
[x]_+=\max(0,x),\quad f_n^\star=2.25,\quad
\mu_p=0.8\mu_{\mathrm{safe}},
\]

\[
h=[\hat f_t+0.02-\mu_p[\hat f_n]_+]_+,\quad
v=\delta p/dt,\quad e_v=v-v_{\mathrm{prev}},
\]

\[
\Delta f_{\mathrm{obs}}=f_k-f_{\mathrm{previous\ fresh}},
\qquad q=50(1+M),
\]

其中 \(M\) 是触觉物体运动指标。按当前默认权重：

\[
\boxed{
\begin{aligned}
J(\delta p)={}&4(\hat f_n-2.25)^2
+10[2-\hat f_n]_+
+8\hat f_t^2
+100h^2\\
&+0.2\|\delta p/dt-v_{\mathrm{prev}}\|^2\\
&+50(1+M)\|\delta f\|^2
+0.5\|\Delta f_{\mathrm{obs}}+\delta f\|^2.
\end{aligned}}
\]

| 目标项 | 作用 |
|---|---|
| 法向平方目标 | 将法向力拉向 2.25 N；2～2.5 N 是软目标范围 |
| 低法向力线性惩罚 | 低于 2 N 时惩罚不足量；当前权重为 10，不是历史讨论中的 100 |
| 切向力平方目标 | 鼓励减小完整切向受力 |
| 提前摩擦惩罚 | 在安全摩擦容量的 80% 附近提前启用，判据还包含 0.02 N 裕量 |
| 速度变化目标 | 抑制相邻位置指令速度变化 |
| 物体稳定代理目标 | 物体观测运动越大，越不愿制造新的预测力变化 |
| 力变化连续目标 | 倾向使下一预测力变化抵消上一实测力变化 |

物体稳定项没有直接预测下一帧物体姿态，而是通过当前运动指标增大力变化惩罚。它是稳定性的代理目标，不等价于真实物体姿态最优控制。

这些项具有不同物理单位，权重决定折中。即使某一轴力为负，最优位移也受法向目标、刚度耦合、摩擦、平滑和力差项共同影响，不能只凭该轴力的正负判断位置应增还是减。

### 6.1 解析梯度

令：

\[
g_n=-K^\top n,\qquad
g_t=-\frac{K^\top\hat{\boldsymbol f}_t}{\max(\hat f_t,10^{-12})}.
\]

用 \(w_n,w_u,w_t,w_f,w_s,w_w\) 表示对应配置权重：

\[
\begin{aligned}
\nabla J={}&2w_n(\hat f_n-f_n^\star)g_n
-w_u g_n\,\mathbf1_{\hat f_n<2}
-2w_tK^\top\hat{\boldsymbol f}_t\\
&+2w_fh\left(g_t-\mu_p g_n\,\mathbf1_{\hat f_n>0}\right)\\
&+\frac{2w_s}{dt}(\delta p/dt-v_{\mathrm{prev}})
-2qK^\top\delta f\\
&-2w_wK^\top(\Delta f_{\mathrm{obs}}+\delta f).
\end{aligned}
\]

线性正部在折点不可微，代码按比较分支选择梯度；恰在低法向力阈值时，该项梯度取零。切向模长接近零时采用微小分母保护。

## 7. 优化约束与求解步骤

### 7.1 归一化和运动约束

令：

\[
s=\min(\text{correction\_limit},v_{\max}dt),\qquad z=\delta p/s.
\]

逐轴边界为：

\[
l_i=\max\left(-1,\frac{v_{\mathrm{prev},i}dt-a_{\max}dt^2}{s}\right),
\quad
u_i=\min\left(1,\frac{v_{\mathrm{prev},i}dt+a_{\max}dt^2}{s}\right).
\]

要求：

\[
l_i\le z_i\le u_i,\qquad 1-z^\top z\ge0.
\]

同时限制位移总范数和逐轴指令加速度；逐轴限制不等于三维加速度总范数限制。

以默认参数和 10 ms 时间步为例，速度对应的单步位移上限是 0.045 mm，比配置的 2 mm 小；从零历史速度开始，加速度边界给出的首步逐轴位移上限为 0.001 mm。

### 7.2 预测接触约束

enforce_cone=True 时，即本轮测量状态已经允许跟踪时，还要求：

\[
\hat f_n\ge0.1,\qquad
\hat f_t+0.02\le\mu_{\mathrm{safe}}\hat f_n.
\]

恢复阶段调用优化时，enforce_cone=False，不加入这两个预测硬约束，但保留全部目标项及运动约束，使位置可以逐步恢复。

必须区分：测量稳定性、wrench 范围及摩擦条件决定是否允许跟踪；SLSQP 约束针对预测力和候选位移。总力/力矩允许范围没有直接写成 SLSQP 预测约束。2～2.5 N 也不是其法向硬区间。

### 7.3 求解流程

1. 以上一指令速度对应的增量为初值，裁剪到逐轴边界。
2. 构造目标的二次部分 \(c+b^\top z+\frac12z^\top Hz\)。
3. 尝试解 \(Hz=-b\) 改善热启动；候选须满足当前约束且完整目标更小。
4. 将低法向力线性项和提前摩擦惩罚加入完整目标。
5. 调用 SciPy 的 SLSQP，提供目标解析梯度和约束雅可比。
6. 在迭代回调检查预算。
7. 求解后独立检查解有限、边界和约束可行、求解器成功且未超预算。
8. 全部通过才返回下一位置；失败返回空位置，step 进入 optimization_failed 保持状态。

objective_before 是原初始热启动候选的目标值，不一定是零增量代价，更不是物理实测误差。物体未稳定而跳过优化时，才明确记录零增量代价，并令前后目标值相同。

## 8. 触觉物体稳定性判断

### 8.1 刚体拟合

_fit_cloud 对参考和当前点云去中心化，以 SVD/Kabsch 求旋转，修正反射，再计算：

\[
\epsilon=
\sqrt{\frac1N\sum_i\|R(p_i-\bar p)-(q_i-\bar q)\|^2}.
\]

返回当前质心、拟合旋转和 RMS 残差。形状不一致、非有限、点数不足或近共线时拒绝拟合。

此方法依赖点索引具有对应关系，不能将任意无对应点云直接当作相同标记序列。

### 8.2 新观测和稳定帧

只有左右两个时间戳都严格更新时，才更新稳定帧和运动历史。每侧计算相邻质心距离 \(d\) 和旋转角变化 \(\theta\)。

任一侧满足下式，即 object_jump=True：

\[
d>d_0+v_o dt_o
\quad\text{或}\quad
\theta>\theta_0+\omega_o dt_o.
\]

两侧都满足更严格的稳定条件，才累计稳定帧：

\[
d\le0.25d_0+v_o dt_o,\qquad
\theta\le0.25\theta_0+\omega_o dt_o.
\]

连续达到 10 帧后 stable_ok=True。重复旧点云不会增加计数。

物体运动指标为两侧之和：

\[
M=\sum_{\text{左右侧}}\left[(d/d_0)^2+(\theta/\theta_0)^2\right].
\]

这些是触觉标记拟合量，可用于检测异常变化；本类没有独立测量物体相对环境的真实滑移。

### 8.3 不稳定时的实际动作

object_jump=True 或 stable_ok=False 时，跳过位置优化，将指令位置设为当前测量位置，内部指令速度设零，禁止轨迹推进。

这一分支没有实现统一的物理减速轨迹，也没有单独冻结四元数。姿态是否冻结取决于此前的摩擦、数据间断或 wrench 恢复门控，不能概括为“任何不稳定都会平滑减速并固定姿态”。

## 9. 摩擦系数更新

### 9.1 初始化与最大观测力比

\[
\mu_{\max,\mathrm{estimate}}=\max(0.18,0.05)=0.18,\qquad
\mu_{\mathrm{safe}}=0.7\times0.18=0.126.
\]

这是先验初始化。在允许更新、双侧新观测、稳定、无突变且 wrench 正常时：

\[
r=\frac{f_t}{\max(|f_n|,10^{-9})},\qquad
\mu_{\max,\mathrm{estimate}}\leftarrow\max(\mu_{\max,\mathrm{estimate}},r).
\]

最近至多五个有效 \(r\) 用于 force_ratio_mean 日志，五帧均值不直接赋给 mu_safe。

更新最大力比并不要求本轮已经满足摩擦锥。friction_ok 在该函数中用于控制后续是否增大 mu_safe，所以稳定且 wrench 正常的锥外样本也能提高最大估值。

最大观测力比不是经过滑移边界标定的真实最大静摩擦系数；稳定接触力比只描述当前受力状态。

### 9.2 先位置恢复，再增大安全系数

摩擦约束不满足，但稳定帧足够、无突变、wrench 正常且外部力控制未运行时，才开始连续摩擦恢复计时。

先冻结首次超界时的实测姿态，通过位置优化尝试恢复。满 10 s 后仍满足更新条件，才允许：

\[
\mu_{\mathrm{safe}}\leftarrow
\min(\mu_{\max,\mathrm{estimate}},1.1\mu_{\mathrm{safe}}).
\]

增长发生于允许更新的新观测调用，并非每秒只增长一次。正常跟踪时不持续增大。

前 10 s 有效摩擦恢复时间从无进展计时中补偿；超过该恢复时段后仍未推进，可能触发 no_progress_timeout。

### 9.3 不稳定后的折减

当前准确条件为：previous_mode 为 tracking，且 not stable_ok，且 not object_jump。满足时：

\[
\mu_{\mathrm{safe}}\leftarrow0.95\mu_{\mathrm{safe}}.
\]

它不覆盖 object_jump=True 分支，也没有用 mu_floor 约束每次衰减后的最小值。

折减发生在本轮跟踪许可和姿态门控形成之后。代码会重算摩擦限额和相关日志，但不会回到前面重新执行本轮全部状态判断。本次没有改变此顺序。

## 10. step 状态机与优先级

step 按以下顺序处理，提前返回会跳过后续步骤：

1. 已锁存故障；
2. wrench PID 恢复是否满 10 s；
3. 调用间断及等待新数据；
4. 数据过期、接触激活、输入有限性；
5. 旋转与力坐标变换；
6. 极限 wrench PID 恢复；
7. 位置指令跟踪偏差；
8. 触觉拟合与稳定性；
9. wrench 范围、摩擦恢复和更新、姿态门控；
10. 无进展超时；
11. 位置优化或不稳定时跳过优化；
12. 推进条件和日志。

极限 wrench 恢复的提前返回在触觉稳定检查之前，不能认为每次返回前已经检查全部条件。

### 10.1 正常跟踪与推进

\[
\mathrm{tracking}=
\neg\mathrm{force\_control}
\land\neg\mathrm{object\_jump}
\land\mathrm{wrench\_range\_ok}
\land\mathrm{friction\_ok}
\land\mathrm{stable\_ok}.
\]

真正推进还要求：

\[
\mathrm{advance}=
\mathrm{tracking}
\land(\mathrm{angle\_error}\le0.08^\circ)
\land(\neg\mathrm{final\_point}\lor\mathrm{settled}),
\]

\[
\mathrm{settled}\iff
\|v_{\mathrm{command}}\|<0.0001\ \mathrm{m/s}.
\]

因此 mode=tracking 与 advance=False 可以同时出现：真实姿态尚未到点，或最后一点的位置调整尚未趋于静止。这里没有原始位置误差的推进门槛。

### 10.2 wrench 范围与 PID 恢复

默认普通范围为：

\[
0.5<\|f\|<10,\qquad 0.05<\|m\|<40.
\]

调用方提供 wrench_in_target 时，跟踪许可采用其结果。

独立的极限恢复条件为：

\[
f_n\le0.1
\quad\text{或}\quad \|f\|>11
\quad\text{或}\quad \|m\|>50.
\]

触发后保存当前姿态，返回 recovery 模式、contact_lost_or_wrench_limit 原因，以及 pid_recovery=True。

本类先返回保存位姿；节点看到请求后用其 PID 恢复函数替换位置输出。当前节点该函数主要沿 Z 调整位置，并保持控制器提供的恢复姿态。

退出要求不再触及极限，且满足任一组合：

\[
0.5<\|m\|<30,\quad2.1<\|f\|<2.5,
\]

或：

\[
18<\|m\|<22,\quad0.7<\|f\|<2.
\]

这些使用当前转换后的模长，并非本类窗口均值。10 s 内未恢复则锁存 wrench_recovery_timeout。

普通范围不满足与极限恢复触发不是同一条件。仅 wrench_in_target=False 会阻止跟踪，本类不会仅凭此布尔值无条件触发上述 PID 请求；需结合调用方 force_control 逻辑理解。

contact_active=False 则直接锁存 contact_inactive，不进入低法向力恢复。

### 10.3 control_gap

调用间隔超过 0.2 s 时进入 waiting_data，不锁存 control_gap。清零内部速度和稳定计数，清除力差基准及摩擦恢复计时，返回保存位姿并禁止推进。

恢复要求：

- 恰好七个时间戳，全部有限；
- 全部晚于本次等待开始时刻；
- 全部不在未来，且年龄不超过 0.2 s；
- 新一次调用间隔本身不再超时。

满足后重新积累稳定性并恢复控制。等待时间从无进展计时中补偿，但不会延长已经启动的 wrench PID 恢复 10 s 上限。

普通 stale_sensor 检查没有这一路径同样的七元素形状检查和全部未来时间戳检查。

### 10.4 hold 和锁存

_hold 返回保存位置、保存旋转及 False，清零内部速度。它不直接发布消息，也不代表机械臂瞬时静止。

| 原因 | 当前行为 |
|---|---|
| control_gap | 等待新数据，非锁存 |
| contact_lost_or_wrench_limit | 请求外部 PID 恢复，非锁存 |
| optimization_failed | 保持当前测量位置对应的指令，后续可以重试 |
| wrench_recovery_timeout | 锁存 |
| stale_sensor | 锁存 |
| contact_inactive | 锁存 |
| nonfinite_input / invalid_quaternion | 锁存 |
| robot_tracking_error | 保存位置指令与实测偏差超过 1 mm，锁存 |
| invalid_object_fit / object_fit_residual | 点云拟合异常，锁存 |
| no_progress_timeout | 有效无进展时间超限，锁存 |

锁存后下次 step 首先返回同一故障。start_segment 不清除 fault，不是解除故障的接口。

## 11. 新轨迹段重置与保留

start_segment 重置：

- 上一调用时刻、最近推进时刻；
- 数据等待、控制间断恢复状态；
- 摩擦恢复计时及摩擦冻结姿态；
- 历史指令速度、连续稳定帧；
- 五帧力比窗口；
- 上一力向量、物体运动指标、力变化模长和向量。

继续保留：

- 当前位姿指令；
- 固定法向、刚度、参考旋转、力偏置、参考点云；
- mu_safe 和历史 mu_max_estimate；
- 上一物体拟合结果及相关时间戳；
- object_unstable；
- 锁存故障；
- wrench PID 恢复计时和冻结姿态；
- 状态字典等未被该方法显式重置的字段。

开始新段并不等于重新创建控制器或清空全部历史。

## 12. 日志解读

| 字段 | 实际含义 |
|---|---|
| normal_force / tangent_force | 变换后的法向力及切向力模长 |
| force_norm / moment_norm | 变换后的合力、力矩模长 |
| normal_in_target | 法向力是否在软目标区间，不是单独的跟踪硬门槛 |
| wrench_in_target | 最终采用的范围判断，可能来自调用方 |
| measured_wrench_in_target | 本类独立计算的范围判断 |
| stable_frames / recovery_frames | 连续稳定新观测数量及阈值 |
| object_jump / stable_ok | 突变标志和稳定恢复结果 |
| friction_limit | \(\mu_{\mathrm{safe}}f_n\)，与 \(f_t+0.02\) 比较 |
| friction_recovery_seconds | 连续满足资格的摩擦恢复时间 |
| friction_adapting | 已完成位置恢复等待但仍需调整摩擦系数 |
| force_ratio_mean | 最近至多五个有效力比均值 |
| mu_max_estimate | 含初始先验的历史最大有效观测力比 |
| tracking_allowed | 允许跟踪，不保证真实姿态到点 |
| advance | 本轮是否真的允许推进 |
| angle_error / angle_error_deg | 当前真实姿态相对于当前目标姿态的误差 |
| linear_speed | 优化位移除以时间步的指令速度模长，不是实测速度 |
| angular_speed | 本实现固定写零，不代表真实角速度 |
| target_angular_rate | 姿态误差除以时间步，不是实测角速度或限速后的独立指令 |
| objective_before / objective_after | 初始候选与最终候选的模型代价 |
| optimized_delta_position | 本周期位置增量，m |
| optimized_next_position | 下一绝对位置指令，m |
| solver_success / solver_message | 综合求解结果及原因 |
| solver_constraint_margin | 所有启用不等式中的最小余量，涉及不同单位，不能统称摩擦裕量 |
| predicted_constraints_enforced | 是否启用预测法向下限及摩擦锥约束 |
| solver_ms / controller_ms | 优化及本类单次控制调用耗时 |
| no_progress_seconds | 经过特定等待/恢复补偿后的无推进时间 |

某些提前返回路径只作 status.update，没有重建全部字段，个别记录可能沿用上轮。应结合 mode、reason、solver_skipped 和 advance 分析。

## 13. 100 Hz 运行条件

100 Hz 对应 10 ms 周期。三维变量、解析梯度和雅可比、二次热启动、30 次迭代上限和 6 ms 求解预算用于降低计算成本。

但设置预算不等于证明实时性：

- 迭代回调是协作式检查，不能中途抢占耗时数值库操作。
- 求解结束再次检查总耗时，超预算解不会被接受。
- controller_budget_exceeded 只记录是否超周期，本身不是中断机制。
- controller_ms 不包括全部 ROS 数据接收、调用方 PID、日志与发布耗时。
- 常规 dt 直接限制在 1～10 ms，实际调用变慢不会使优化步长无限增加。
- 点云规模、数值库线程、系统负载和控制调度会影响实际频率。

应在实际环境统计控制耗时中位数、P95、P99、最大值、超预算比例，以及包含 ROS 调度的真实调用周期，再判断是否达到 100 Hz。本次仅添加注释和文档，没有运行机械臂或重新执行性能基准测试。

## 14. 本次修改和验证

所有非空行均补充中文解释，保留空行用于分段。原多行类文档改为等值相邻字符串，以便每行加注释，类的 __doc__ 文本完全不变。

验证结果：

1. 比较修改前后完整 Python 抽象语法树，忽略行列位置，结果一致；比较包含文档字符串、默认值、常量、条件和表达式。
2. Python 编译接口语法检查通过。
3. tokenize 检查确认 618 个非空行均有中文注释。

此次没有修改本文指出的行为差异或局限，没有调整规划节点或测试代码。

