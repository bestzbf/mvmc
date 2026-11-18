# 魔搭默认权重：195500

[项目导航](project-guide.md) · [魔搭模型卡](https://modelscope.cn/models/binfengzou/MultiD4RT) · [研究推理准备](research-inference.md)

默认文件由魔搭仓库的 `configuration.json` 指定。**195500 含 TrajectoryRefiner，使用随权重发布的研究推理代码。** 顶层 `mvmc infer` 和 `MVMC.from_checkpoint()` 继续对应历史 60k 权重。

## 下载并准备样例

```bash
python -m pip install 'modelscope>=1.35.1'
python scripts/download_default_checkpoint.py
python -m pip install -r weights/mvmc-default/runtime/requirements.txt
```

从仓库根目录执行。下载脚本读取魔搭的默认配置、校验权重和源码的 SHA-256，解包到 `weights/mvmc-default/runtime/`，并生成无需额外数据的双相机 RGB 小样例。可用 `--output` 自定义下载目录，随后相应替换下文路径。

## 运行推理

```bash
PYTHONPATH=weights/mvmc-default/runtime python -m multicamera4rc.infer \
  --manifest weights/mvmc-default/demo/observations.json \
  --checkpoint weights/mvmc-default/mvmc_giant_step195500.pt \
  --output outputs/default_demo --device cuda --amp --amp-dtype bfloat16
```

没有 CUDA 可用 `--device cpu` 并省略 AMP 参数。样例用于确认权重能加载、推理能完成。用自己的图像时，按[清单格式](inference.md#图像与-manifest)准备同分辨率、共用帧号的 RGB 序列并替换 `--manifest`。模型不读取相机或深度真值；保存的 K/V 与时间原点保留为 `target_time`、`clip_start`。

文件约 6.11 GB（FP32）；SHA-256 是 `2d76af9a0ded8b90f93954d1c730bd85a874e8d99bcae43149b7d2088cd4c5ea`。导出保留全部 **1148 个张量**，不包含训练状态。默认版本按 DexYCB 10、Panoptic 6、MV-Kubric 30 共 46 个完整评估场景选择；三组等权 APD **87.58%**、EPE **14.14 cm**。这不代表同一权重在每个单项均最优。
