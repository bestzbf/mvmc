# 研究训练与基准评估

[项目代码导航](../docs/project-guide.md) · [首次运行训练](training/source/README.md) · [研究权重推理](../docs/research-inference.md)

本目录保存已导出的实验训练实现、step134400 冻结模型，以及全量评估的准备、推理、评分、恢复和报告代码。三个目录分别记录来源版本和文件哈希。训练核心文件与部署版本的核验、decoder K/V 范围、refiner 和后续恢复器的边界见[训练模型说明](../docs/training-architecture.md)。

| 目录 | 内容 |
| --- | --- |
| [training](training/README.md) | 研究训练源码，以及以 `069b1c2` 为基础的历史六卡部署快照（v22 源码覆盖、v23 控制器及配置模板） |
| [frozen_134400](frozen_134400/README.md) | `dd45a946` 的冻结模型源码、来源清单和独立的导出身份校验 |
| [evaluation](evaluation/README.md) | 实际使用的评估调度、模型适配、评分、显存恢复和报告脚本，以及外部依赖版本 |

数据范围和指标说明见[评估结果索引](results/README.md)。

## 使用研究代码

训练与冻结评估各自加载对应的 `multicamera4rc`，使用独立的 Python 进程和运行目录。上表中的入口说明列出所需 Python 环境、数据清单、模型权重和第三方源码版本。复现某个旧实验时先根据其身份记录选版本。

训练启动器与评估脚本的机器绑定通过 `materialize.py` 渲染。将对应的 `bindings.example.json` 复制到工作区外，填写本机路径和设备 UUID。绑定值使用无空格和引号的字面量。`${RUNTIME_ROOT}` 会展开为输出目录的绝对路径。

```bash
python research/materialize.py research/evaluation \
  /work/multid4rt-bindings/evaluation.json /work/multid4rt-runtime/evaluation
```

渲染器校验模板哈希，按声明重新绑定脚本之间的源码哈希，并写出实际文件的哈希清单。训练模板的渲染顺序见[训练入口说明](training/README.md)。评估队列和评分配方根据绑定后的数据与预测来源重新生成。使用本仓库的冻结模型副本启动新评估时，按[冻结模型说明](frozen_134400/README.md)接入导出身份校验，并使用独立的预测目录。

训练配置、完整代码身份和 GPU 运行环境共同决定能否继续已有训练状态。复用历史 checkpoint、预测和评分时，沿用原始记录要求的源码身份、数据绑定和校验清单。

## 回归测试

- [训练回归测试](training/tests/activation/README.md)覆盖主训练、CPU 分布式、六卡启动、检查点发布和训练侧评估准入。
- [评估回归测试](evaluation/tests/README.md)覆盖调度、六种模型适配、评分、恢复和报表。运行器在临时目录部署代码并使用合成输入。
