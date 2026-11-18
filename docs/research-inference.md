# 用研究 checkpoint 推理

[项目导航](project-guide.md) · [研究训练](../research/training/source/README.md) · [公开 60k 推理](inference.md)

研究训练 checkpoint 可能含 `TrajectoryRefiner`。用与其匹配的 `multicamera4rc` 源码，在独立环境启动。需要复跑冻结正式评估时按[对应快照](../research/frozen_134400/README.md)执行。

## 准备图片或视频

准备 `input.json`。下面的两台相机共用帧号 0 和 5；路径相对于这个配置文件。

```json
{
  "clips": [{
    "id": "two_cameras",
    "cameras": [
      {"camera_id": 0, "images": [{"image": "cam0_000.png", "time_idx": 0}, {"image": "cam0_005.png", "time_idx": 5}]},
      {"camera_id": 1, "images": [{"image": "cam1_000.png", "time_idx": 0}, {"image": "cam1_005.png", "time_idx": 5}]}
    ]
  }]
}
```

读取视频时，把相机的 `images` 换成 `"video": "cam1.mp4", "frame_indices": [12, 17], "time_idx": [0, 5]`。前者是文件内帧号，后者是多相机共享时间标签。工具不负责同步或去畸变；它保留像素分辨率，只在右侧和底部补到共同的 patch14 画布。清单中的第 0 张观察必须有效，并定义参考坐标。

## 运行和导出

在已安装[匹配研究源码](../research/training/source/README.md)的环境执行。填写自己可读的配置、权重和新输出目录。

```bash
INPUT_CONFIG=/path/to/input.json
CHECKPOINT=/path/to/research_checkpoint.pt
RUN_DIR=/path/to/new_inference_run

python -m multicamera4rc prepare-inference \
  --config "$INPUT_CONFIG" --output "$RUN_DIR/prepared"

python -m multicamera4rc infer \
  --manifest "$RUN_DIR/prepared/clips.json" --checkpoint "$CHECKPOINT" \
  --device cuda --amp --amp-dtype bfloat16 --output "$RUN_DIR/predictions"

python -m multicamera4rc export \
  --predictions "$RUN_DIR/predictions" --output "$RUN_DIR/ply"
```

`--source-obs` 指观察列表下标，`--query-times` 指时间标签。未显式传源时优先使用清单中的每片段源选择；准备工具默认选每台相机最早的一张有效图。默认沿用 checkpoint 的 decoder K/V 和时间原点。显式覆盖它们会改变计算，需把结果作为独立实验。

预测目录中的 `inference.json` 记录有效配置，`clip_000000.pt` 等保存每段的预测。PLY 导出使用原图区域，默认保留有效点；如果只是预览，可设 `--max-points 200000` 限制每时刻点数。研究推理的 `--data-config` 是可选的数据适配器入口，与 `--manifest` 互斥；做 RGB-only 评估时使用协议要求的冻结 manifest。
