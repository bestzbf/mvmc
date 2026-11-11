# mvmc

**Multi-camera 4D Reconstruction and Tracking**

mvmc（原 MultiD4RT）从同步多相机 RGB 序列重建随时间变化的三维场景，输出稠密几何、scene flow、三维轨迹和逐相机可见性。输入只需要图像、相机身份和共享帧号，无需已知内外参或 timestamp。相机数量与序列长度可以变化。

项目已更名为 `mvmc`，Python 使用 `from mvmc import MVMC, MVMCConfig`，命令行为 `mvmc`。原 `multid4rt` 包、类名和命令继续兼容；已有权重文件与魔搭仓库仍使用原名称。

## 模型版本与 K/V 范围

| 使用方式 | decoder 直接 K/V | refiner |
| --- | --- | --- |
| 新建 `MVMCConfig()` | 默认 `all_observations`（全部有效输入观察） | 无 |
| 公开 60k giant 权重 | `target_time`（目标时刻全部有效相机） | 无 |
| 当前研究训练 | `target_time`（目标时刻全部有效相机） | 有，见研究源码 |

encoder 在两种模式中都融合全部有效相机与时刻；这不代表 decoder 直接使用全部帧 K/V。恢复权重应使用保存的配置，不能凭新建配置的默认值判断训练设置。顶层推理包不支持带 refiner 的研究 checkpoint。完整区别、refiner 输入输出和源码核验见[训练模型说明](docs/training-architecture.md)。

## 训练与全量评估

训练核心源码、导出时的训练控制器、全量评估流程和 step134400 冻结模型源码随主分支提供。研究入口、源码版本及部署配置见[训练与评估说明](research/README.md)；后续运维恢复器与该导出快照的区别见[训练模型说明](docs/training-architecture.md)。

评估结果分别记录在[子集评估表](research/results/subset.md)和[全量评估表](research/results/full.md)。

## 安装

```bash
git clone https://github.com/bestzbf/mvmc.git
cd mvmc
python -m pip install -e '.[weights,viz]'
mvmc --help
```

## 预训练权重

[魔搭模型仓库：binfengzou/MultiD4RT](https://modelscope.cn/models/binfengzou/MultiD4RT) 提供 **60k giant** 推理权重，适用于多相机 4D、多视角单时刻与单目动态输入。

| 项目 | 内容 |
|---|---|
| 权重文件 | `multid4rt_giant_step60000.pt` |
| 模型 | giant：patch14、1536 维、40 层 encoder |
| 格式 / 大小 | 原始 FP32，6.11 GB；包含模型参数与配置 |
| 兼容版本 | MultiD4RT v0.5.1 |
| SHA256 | `1161d6bfaa0d277b22108c5b9e803b9e1c36e289f98799e35bd59a9b821197e0` |

下载到项目的 `weights/` 目录：

```bash
python -m pip install 'modelscope>=1.35.1'
```

```python
from modelscope.hub.file_download import model_file_download

checkpoint = model_file_download(
    'binfengzou/MultiD4RT',
    file_path='multid4rt_giant_step60000.pt',
    local_dir='weights',
)
print(checkpoint)
```

加载真实权重或直接运行可视化：

```python
from mvmc import MVMC

model = MVMC.from_checkpoint('weights/multid4rt_giant_step60000.pt').eval().cuda()
```

```bash
mvmc visualize --input camera_sequences/ --mode multicamera --checkpoint weights/multid4rt_giant_step60000.pt --output outputs/multicamera --device cuda --amp --amp-dtype bfloat16
```

`from_checkpoint()` 自动恢复权重保存的配置。该 60k 权重使用 `decoder_memory=target_time`、`time_origin=clip_start`：encoder 融合全部输入观察，运动 decoder 的 KV 来自目标时刻的有效相机。复现该权重的输出时保持这些设置。魔搭仓库附模型配置、校验值和导出验证记录；1,106 个状态张量与原始 checkpoint 完全一致，并通过真实双相机 RGB 的 CUDA 推理检查。

## 模型输入与输出

同步图像为 `[B,C,T,3,H,W]`，按相机优先展平为 `[B,S,3,H,W]`，其中 `S=C*T`。RGB 使用 `[0,1]` 浮点值或 uint8。保持图像比例，将高宽准备为 14 的倍数且均至少 28；基础 `infer` 命令默认不缩放。

- `camera_id [B,S]`：同一物理相机共享 ID，片段内一致即可。
- `time_idx [B,S]`：同步相机共享帧号，保留抽帧间隔，例如 `0,5,10`。
- `observation_valid [B,S]`：可选，用于屏蔽缺失观察。
- `source_obs_idx [B,Q]`：选择哪些输入图作为轨迹的源图。
- `query_time_idx [B,Tq]`：请求输出的目标帧号。

例如 C2T3：

```text
观察下标     0   1   2   3   4   5
camera_id    0   0   0   1   1   1
time_idx     0   5  10   0   5  10
```

| 输出 | 维度 | 含义 |
|---|---|---|
| `depth` | `[B,S,H,W]` | 每张输入图的深度 |
| `K` / `c2w` | `[B,S,3,3]` / `[B,S,4,4]` | 预测内参与相机位姿 |
| `base_points` | `[B,S,H,W,3]` | 统一世界坐标中的基础点 |
| `trajectory` | `[B,Q,Tq,H,W,3]` | 源图每个像素的三维轨迹 |
| `displacement` | `[B,Q,Tq,H,W,3]` | 相对基础点的三维位移 |
| `visibility` | `[B,Q,Tq,C,H,W]` | 每个目标时刻、相机中的可见性 |

## 模型结构

共享 encoder 先提取每张图的局部特征，再融合全部相机与时刻。相机身份通过同相机 attention bias 编码；共享帧编码和 time token 提供时间信息。

几何分支预测深度、内参和位姿，反投影得到基础点 `P`。新建模型的运动 decoder 默认使用全部输入观察的 patch 特征作为 KV，以目标时间和源到目标的时间差进行 AdaLN 调制，预测三维位移 `Δ`；加载 checkpoint 时使用其保存的 KV 范围。最终轨迹为 `X(t)=P+Δ(t)`，源时刻严格满足 `Δ(t_source)=0`。可见性分支结合轨迹投影与目标图像特征输出逐相机可见性。

giant 配置使用 patch14、1536 维 encoder 和 40 层；完整模块与中间张量维度见 [模型结构](docs/architecture.md)。

## Python 接口

```python
import torch
from mvmc import MVMC, MVMCConfig, pack_synchronized_images

model = MVMC(MVMCConfig.tiny()).eval()
rgb = torch.rand(1, 2, 3, 3, 28, 42)
inputs = pack_synchronized_images(rgb, frame_indices=[0, 5, 10])

with torch.inference_mode():
    pred = model(**inputs, source_obs_idx=[0, 3])

print(pred['trajectory'].shape)  # [1,2,3,28,42,3]
print(pred['visibility'].shape)  # [1,2,3,2,28,42]
```

tiny 示例使用随机权重检查接口。实际推理加载对应 checkpoint：

```python
model = MVMC.from_checkpoint('model.pt').eval()
```

`model.encode()` 的结果可供同一输入下多个源图/目标时间请求复用。`mvmc.tasks.scene_flow` 提取同一物理点的世界位移，`sample_tracks` 查询源图像素的三维轨迹。

## 命令行推理

准备图片与相机/时间标签组成的 JSON 或 JSONL manifest，然后运行：

```bash
mvmc infer --manifest observations.json --checkpoint model.pt --output outputs/prediction --device cuda --amp --amp-dtype bfloat16
```

输出 `inference.json` 和各片段的 `.pt` 预测结果。manifest 格式、源图选择、目标帧与 checkpoint 配置见 [推理说明](docs/inference.md)。

```bash
mvmc demo --output outputs/demo
```

CPU demo 用随机 tiny 权重验证输入输出流程。

## 推理可视化

同一个命令支持三种输入：

```bash
# 多相机 4D：每个相机一个子目录，使用共享帧号
mvmc visualize --input camera_sequences/ --mode multicamera --checkpoint model.pt --output outputs/multicamera --device cuda --amp

# 多视角单时刻：一个目录中的图片代表不同相机
mvmc visualize --input scene_views/ --mode multiview --checkpoint model.pt --output outputs/multiview --device cuda --amp

# 单相机动态重建 / 追踪：视频或连续帧目录
mvmc visualize --input video.mp4 --mode monocular --checkpoint model.pt --output outputs/monocular --device cuda --amp
```

打开输出目录中的 `viewer.html`，可以旋转三维场景、播放时间、切换相机、查看 RGB/深度和点选三维轨迹。页面离线可用，另导出 PLY、轨迹与 scene flow。可视化默认最多取 16 帧、等比例缩小至长边 518，使用 `--max-frames 0 --long-side 0` 保留全部帧和原始尺寸；补边像素会从显示与导出中排除。

没有 checkpoint 时可先检查界面：

```bash
mvmc visualize --demo --mode multicamera --output outputs/viewer_demo --device cpu
```

此示例使用随机 tiny 权重，仅验证流程。文件夹格式、视频输入、原图像素查询、预测复用和导出维度见 [可视化说明](docs/visualization.md)。

## 真实数据实例

打开 [examples/index.html](examples/index.html) 可查看 step 56,000 giant 权重生成的五组精选重建：两组 Syn4D C2T4 人物动态场景、BlendedMVS C8T1 城市建筑、Hypersim C4T1 室内与 PointOdyssey C1T16 人物序列。每组附原始 RGB、交互式三维页面、PLY、轨迹与 scene flow，可直接离线打开。案例按展示质量筛选；具体输入与重新推理命令见[实例说明](examples/README.md)。

| Syn4D · Temple 双相机动态 | Syn4D · Warehouse 双相机动态 |
|---|---|
| [![Syn4D Temple](examples/syn4d_temple_c2t4/preview.png)](examples/syn4d_temple_c2t4/viewer.html) | [![Syn4D Warehouse](examples/syn4d_warehouse_c2t4/preview.png)](examples/syn4d_warehouse_c2t4/viewer.html) |
| **BlendedMVS · 城市建筑** | **Hypersim · 室内重建** |
| [![BlendedMVS](examples/blended_city_c8t1/preview.png)](examples/blended_city_c8t1/viewer.html) | [![Hypersim](examples/hypersim_c4t1/preview.png)](examples/hypersim_c4t1/viewer.html) |
| **PointOdyssey · 单相机人物追踪** | |
| [![PointOdyssey](examples/pointodyssey_c1t16/preview.png)](examples/pointodyssey_c1t16/viewer.html) | |

也可下载[完整实例包](https://github.com/bestzbf/mvmc/releases/download/v0.5.1/multid4rt-real-examples-step56000.zip)，解压后打开 `examples/index.html`。GitHub 文件预览显示源码，交互页面请在本地浏览器中打开。

组件来源与许可见 [NOTICE](NOTICE)。
