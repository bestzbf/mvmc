# mvmc 代码导航

[项目首页](../README.md) · [公开推理](inference.md) · [研究训练](../research/training/source/README.md) · [研究权重推理](research-inference.md) · [基准评估](../research/evaluation/README.md)

先按权重选择代码，再选择命令。公开推理、研究训练和冻结评估共享项目设计，但有各自的模型实现和运行依赖。

| 要做什么 | 入口 | 权重或其他输入 |
|---|---|---|
| 用魔搭默认 195500 giant 权重推理 | [默认权重的下载和推理](default-checkpoint.md) | 权重与配套运行时，含 TrajectoryRefiner |
| 用兼容的 60k giant 权重运行自己的图像 | `mvmc infer` / [公开推理指南](inference.md) | 60k 权重 + RGB manifest；公开模型不含 TrajectoryRefiner |
| 浏览单目、多视角或多相机视频的输出 | `mvmc visualize` / [可视化指南](visualization.md) | 公开推理兼容权重，或用 `--demo` 验证流程 |
| 新训练、续训或调试 GT | `python -m multicamera4rc launch` / [训练入口](../research/training/source/README.md) | 数据配置、损失配置 + 初始化权重或完整断点 |
| 用研究训练权重推理 | `python -m multicamera4rc prepare-inference` → `infer` → `export` / [研究推理指南](research-inference.md) | 与训练 checkpoint 匹配的源码和配置；保留 refiner |
| 复算正式指标 | [冻结模型](../research/frozen_134400/README.md) + [评估队列](../research/evaluation/README.md) | 绑定的模型源码、checkpoint、固定输入清单、评分协议和外部方法环境 |

## 仓库结构

| 路径 | 负责什么 |
|---|---|
| [`mvmc/`](../mvmc) | 公开包名和 CLI；导出 `MVMC`、`MVMCConfig` |
| [`multid4rt/`](../multid4rt) | 公开模型、读图、权重加载、推理、轨迹采样和可视化；保留旧包名兼容 |
| [`research/training/source/multicamera4rc/`](../research/training/source/multicamera4rc) | 研究模型、训练器、数据适配器、训练权重推理和点云导出 |
| [`research/training/`](../research/training) | 历史训练部署快照、启动与验证模板、源码身份和回归测试 |
| [`research/evaluation/`](../research/evaluation) | 固定评估任务的准备、预测、评分、恢复和报表；脚本和路径经 `materialize.py` 绑定 |
| [`research/frozen_134400/`](../research/frozen_134400) | 134400 步权重对应的冻结模型源码和验证清单 |
| [`research/results/`](../research/results) | 已发布评估表及口径说明 |
| [`tests/`](../tests)、[`examples/`](../examples) | 公开包回归测试和可直接浏览的示例输出 |

## 训练：从数据到检查点

1. [`launch.py`](../research/training/source/multicamera4rc/launch.py) 分配设备并启动训练；[`train.py`](../research/training/source/multicamera4rc/train.py) 负责组批、前向、反向、断点发布。设备号指物理 GPU，下发给训练进程后由运行环境重新映射。
2. [`datasets/mixture.py`](../research/training/source/multicamera4rc/datasets/mixture.py) 根据数据配置选择数据源、片段与相机布局；[`datasets/adapters/`](../research/training/source/multicamera4rc/datasets/adapters) 负责解码原始文件与读标注。同一数据集可以有不同监督配方。
3. [`datasets/bridge.py`](../research/training/source/multicamera4rc/datasets/bridge.py) 统一帧轴、图像像素和相机坐标；[`dense.py`](../research/training/source/multicamera4rc/datasets/dense.py)、[`partial_dense.py`](../research/training/source/multicamera4rc/datasets/partial_dense.py)、[`camera_only.py`](../research/training/source/multicamera4rc/datasets/camera_only.py) 和其他监督模块生成最终标签及掩码。原始文件有字段，只代表标签可能可用；有效掩码决定哪些位置进入损失。
4. [`model.py`](../research/training/source/multicamera4rc/model.py) 预测几何和运动，按保存的配置启用 [`refinement.py`](../research/training/source/multicamera4rc/refinement.py)。[`losses.py`](../research/training/source/multicamera4rc/losses.py) 分开处理深度、相机、稠密和稀疏轨迹、可见性；[`optim.py`](../research/training/source/multicamera4rc/optim.py) 与 [`schedule.py`](../research/training/source/multicamera4rc/schedule.py) 管理参数组与学习率。
5. 训练输出包括完整检查点和 `latest.pt`；使用 [`checkpoint.py`](../research/training/source/multicamera4rc/checkpoint.py) 保存／读取。生产续训还依赖实际数据与损失配置、优化器状态、学习率计划、随机状态、采样进度和绑定环境。训练目标步数与学习率计划终点是两项配置。

## 推理：从 RGB 到轨迹

输入为同步 RGB、`camera_id`、共享整数 `time_idx` 以及有效观察掩码。相机内外参由模型预测。几何分支给出三维起点，动态分支给出相对位移，两者合成轨迹。研究模型可在可见性预测前细化轨迹。[张量与结构](architecture.md) · [K/V 与 refiner 的版本区别](training-architecture.md)

- **公开包**：[`io.py`](../multid4rt/io.py) → [`infer.py`](../multid4rt/infer.py) → [`model.py`](../multid4rt/model.py)。[`tasks.py`](../multid4rt/tasks.py) 从稠密输出提取 scene flow 和查询轨迹；[`visualize.py`](../multid4rt/visualize.py) 生成可视化。
- **研究包**：[`prepare_inference.py`](../research/training/source/multicamera4rc/prepare_inference.py) 把图像或对齐后的视频片段转成统一清单；[`infer.py`](../research/training/source/multicamera4rc/infer.py) 恢复研究模型；[`export.py`](../research/training/source/multicamera4rc/export.py) 从预测导出 PLY。
- **正式评估**：相同名字的 `multicamera4rc` 可能来自多个源码树，使用独立进程，并记录加载文件、权重和数据清单哈希。便捷的可视化入口不会自动执行正式评估协议。

部署快照、自定义数据、外部模型权重与运行日志各有独立身份。[研究代码与部署说明](../research/README.md)说明如何绑定。要确认正在跑的训练，应查正在执行的启动参数、实际源码和配置哈希、最近更新与检查点；历史 README 或仓库默认配置不能代替运行状态。
