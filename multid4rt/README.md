# mvmc

Multi-camera 4D Reconstruction and Tracking.

本目录是兼容公开 60k 权重的推理实现，保留 `multid4rt` 导入名，也由 `mvmc` 导出。它不含研究训练版本的 `TrajectoryRefiner`；两种 decoder K/V 模式、实际权重配置和加载边界见[训练模型说明](../docs/training-architecture.md)。

[项目入口](../README.md) · [模型结构](../docs/architecture.md) · [推理说明](../docs/inference.md) · [可视化](../docs/visualization.md)

```python
from mvmc import MVMC, MVMCConfig, pack_synchronized_images
```

输入 RGB、相机 ID 和共享帧号；输出深度、相机参数、世界基础点、三维轨迹、位移与逐相机可见性。命令行为 `mvmc infer`、`mvmc visualize` 和 `mvmc demo`。
