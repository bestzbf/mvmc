# mvmc 训练模型与公开推理接口

本页区分三种版本，避免将配置默认值、已训练权重与冻结评估混为一谈。

| 版本 | 直接 decoder K/V | TrajectoryRefiner | 加载位置 |
| --- | --- | --- | --- |
| 顶层新建 `MVMCConfig()` / `MultiD4RTConfig()` | 默认 `all_observations`，可显式选 `target_time` | 无 | `mvmc` / `multid4rt` |
| 公开 60k giant 权重 | `target_time`，`time_origin=clip_start` | 无 | 顶层 `from_checkpoint()` |
| 当前研究训练 | 启动参数明确为 `target_time` | 有，具体尺寸与开关以 checkpoint 配置为准 | `research/training/source/multicamera4rc` |
| step134400 冻结评估 | 以该冻结 checkpoint 配置为准 | 使用冻结源码实现 | `research/frozen_134400/model_source` |

`mvmc` 是项目新名称。顶层 Python 接口兼容原 `multid4rt`，研究命名空间仍为 `multicamera4rc`；名称相似不表示这些版本的模型配置可以互换。已有 checkpoint 的参数键、架构标记、外部权重文件名和历史结果身份保留原值。

## encoder 与动态 decoder

encoder 对全部有效输入观察进行跨相机、跨时刻融合。动态 decoder 使用源图 patch 作为查询，并加入目标时刻和相对源时刻条件。

- `target_time`：每个目标时刻读取该时刻所有有效相机的 patch。若有 C 台相机、每图 L 个 patch，每个目标的 K/V 长度为 C×L（无效相机被 mask）。其他时刻的信息可以经 encoder 间接进入这些 patch。
- `all_observations`：每个目标直接读取全部输入观察的 patch。完整 C 相机、T 时刻输入的 K/V 长度为 C×T×L。

两种模式共用参数形状，但改变直接注意力范围。加载同一组参数不代表输出或质量相同。当前训练使用 `target_time` 是实际配置，不能仅凭它不是构造器默认值就认定为实现错误；改为全帧 K/V 属于需要重新验证的模型变更。

## 训练中的 TrajectoryRefiner

实现：[refinement.py](../research/training/source/multicamera4rc/refinement.py)，调用顺序：[model.py](../research/training/source/multicamera4rc/model.py)。它位于动态 decoder / MotionDPT 之后、VisibilityHead 之前。

输入包括：初始世界坐标轨迹、MotionDPT 特征、投影后的源与目标图像特征、预测深度及深度置信度、预测 K 和 c2w、有效观察/时间 mask、同相机关系、帧号、源时刻槽位和场景尺度。它不接收 GT 几何，也**不接收 VisibilityHead 输出的可见性概率**。投影是否有效、局部匹配支持和深度线索均不等同于遮挡真值。

refiner 将轨迹与运动特征降到 patch 网格。在当前轨迹投影附近的多尺度目标特征窗口中计算相关性，融合几何线索；启用时间模块时，对同一个源点跨时刻的隐藏状态执行 self-attention，再通过共享更新网络迭代预测世界 XYZ 修正。这里的局部匹配和时间 self-attention 与前面动态 decoder 的 cross-attention K/V 是不同操作。启用 refiner 时内部处理全部有效观测时刻，最后再选择用户请求的目标时刻。

每轮修正先经过 `tanh` 并乘场景尺度和配置限幅，再减去源时刻修正，保持源时刻位移为零。可学习凸上采样将低分辨率修正恢复为 `[B,U,H,W,3]`，加到原有稠密位移上；它不会用上采样的粗轨迹替换原始稠密预测。模块还返回匹配比例和相对修正幅度诊断量。

匹配坐标、深度/相机几何线索及轨迹状态编码使用停止梯度；图像特征、运动特征与更新网络仍可学习，原轨迹的残差相加路径仍保留梯度。模块不直接更新 depth、K 或 c2w。VisibilityHead 随后基于修正后的轨迹预测逐相机可见性。

## 源码核验与复现边界

本次仓库审计逐一比较了部署源码与仓库 `research/training/source/multicamera4rc/` 中五个核心文件，SHA-256 一致：

| 文件 | SHA-256 |
| --- | --- |
| `model.py` | `a63a8802d267477c49230ddc14e19da2325651b56b10a1ee3e8ffff388619833` |
| `config.py` | `0869f8ad0ca6a2d9102bd3989cc38ac60f85e9e3270f7059762327287ed180af` |
| `refinement.py` | `5bbb237e0d6c59804e783bc26076ec12f56babeafc835374a93347c9da5a1a70` |
| `train.py` | `ab7051ed4154493acc7c94a9c97b28d021f601770c7bbb4d5b13282cdc031072` |
| `launch.py` | `8c2993f1d46b106f18d61bd6eccbc0af332c4a3c490b67562e94148c7149d7f6` |

这证明五个核心文件一致，不表示仓库包含最新运维恢复器、私有数据绑定、运行环境或训练权重。训练目录的 v23 supervisor 是导出时快照；后续恢复使用的 `reliability_hardened_v3/resume_latest.py` 不在该导出中。当前恢复从 step175300 继续，目标为 200000。复现实验还需匹配 checkpoint 配置、数据清单、损失配置与评估协议；历史结果不因项目更名而变成当前训练结果。
