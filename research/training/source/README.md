# MultiCamera4RC 研究训练

[项目代码导航](../../../docs/project-guide.md) · [部署与续训](../README.md) · [研究权重推理](../../../docs/research-inference.md) · [基准评估](../../evaluation/README.md)

这里是 mvmc 的研究代码。Python 包为 `multicamera4rc`，包含训练、GT 加载、研究模型和配套推理。它和顶层公开推理包分别安装。冻结评估也有一套独立的 `multicamera4rc`，两套研究源码不要导入同一个进程。

## 安装与 CPU 验证

从本仓库根目录执行。真实训练需要相应数据、匹配的初始化权重或完整断点，以及可用的训练 GPU。

```bash
python -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -e './research/training/source[data,test,weights]'

python -m multicamera4rc --help
python -m multicamera4rc demo --output outputs/research_smoke --train-steps 2
python -m multicamera4rc adapter-demo --output outputs/research_adapter_smoke --train-steps 2
(cd research/training/source && python -m pytest -q)
```

`demo` 使用生成样本跑小模型训练和推理；`adapter-demo` 另验证原生 HDF5 适配器。两者检查执行路径，不用于衡量模型精度。部署级准入、分布式和断点回归见[训练测试指南](../tests/activation/README.md)。

## 配置与启动

[原生 Kubric 配置](multicamera4rc/configs/kubric_multicam_native.yaml)和[混合配置示例](multicamera4rc/configs/mixed_native.yaml)用来说明字段。将数据配置复制到自己的运行目录并替换机器路径，核对 split、场景清单、每源相机／时间布局与 `dense_kind`。模板不代表当前训练采样配方。

```bash
# 新实验：选择已确认空闲的 GPU 和新输出目录。
python -m multicamera4rc launch \
  --gpus 0 --preset giant --batch-size 1 --workers 2 \
  --data-config /path/to/train_data.yaml --pretrained /path/to/4RC/model.safetensors \
  --decoder-memory target_time --output /path/to/new_training_run \
  --steps 1000 --save-every 100
```

- 新初始化用 `--pretrained`，续训用 `--resume`；`--steps` 是总目标步数。真正续训还要恢复模型配置、Adam 状态、学习率、随机状态和采样游标。不要用裸权重冒充训练断点。
- 实验的损失和非默认超参放在可归档的配置里。传 `--loss-config` 时，保存实际配置和哈希。已有 checkpoint 的 K/V、时间原点和 refiner 设置应以保存值为准。
- `launch --dry-run` 用于检查命令和设备映射。共享 GPU 和生产续训需要对应的[绑定配置、运行环境与部署验证](../README.md)，不应直接用上面的新实验命令替代控制器。

## 代码分工和监督

| 要查什么 | 从这些文件开始 |
|---|---|
| 启动、单步逻辑、参数组 | [launch.py](multicamera4rc/launch.py)、[train.py](multicamera4rc/train.py)、[optim.py](multicamera4rc/optim.py)、[schedule.py](multicamera4rc/schedule.py) |
| 数据如何选入 | [mixture.py](multicamera4rc/datasets/mixture.py)、[registry.py](multicamera4rc/datasets/registry.py)、[adapters/](multicamera4rc/datasets/adapters) |
| RGB、标定和标签如何对齐 | [bridge.py](multicamera4rc/datasets/bridge.py)、[augmentation.py](multicamera4rc/datasets/augmentation.py) |
| 哪些标签进入 loss | [dense.py](multicamera4rc/datasets/dense.py)、[partial_dense.py](multicamera4rc/datasets/partial_dense.py)、[camera_only.py](multicamera4rc/datasets/camera_only.py)、[kinoptic_human.py](multicamera4rc/datasets/kinoptic_human.py)、[losses.py](multicamera4rc/losses.py) |
| 模型、可见性和 refiner | [model.py](multicamera4rc/model.py)、[motion.py](multicamera4rc/motion.py)、[refinement.py](multicamera4rc/refinement.py)、[visibility.py](multicamera4rc/visibility.py) |
| 保存、恢复和推理 | [checkpoint.py](multicamera4rc/checkpoint.py)、[prepare_inference.py](multicamera4rc/prepare_inference.py)、[infer.py](multicamera4rc/infer.py)、[export.py](multicamera4rc/export.py) |

调试某个数据源时依次检查原始标注、适配器输出及最终有效掩码。几何配方只监督有深度的同一时刻起点；未知的跨帧运动不填成静止。轨迹、位移、相机、可见性和人体关节分别有有效性条件。Stereo4D 提供稀疏伪轨迹，Kinoptic 实测深度、官方关节位移和表面伪位移分别装配。实际启用的数据和权重以绑定的生产配置为准。文档里的历史 GT 资格和训练状态不代替活跃日志。

模型以 RGB、相机 ID 和共享整数帧号为输入；GT 相机与深度是监督。公开接口、研究模型和冻结 checkpoint 的关系见[版本与网络结构](../../../docs/training-architecture.md)。

## 来源与许可

沿用上游 4RC 的部分模块和参数命名。第三方模块、修改和许可见 [NOTICE](multicamera4rc/vendor/NOTICE)、[来源表](multicamera4rc/vendor/UPSTREAM.json)、[S-Lab 许可](multicamera4rc/vendor/LICENSE.S-Lab)和 [Apache-2.0](multicamera4rc/vendor/LICENSE.Apache-2.0)。
