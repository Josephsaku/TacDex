# TacDex13：单片 ID3 的触觉仿真迁移

本仓库负责**离线数据输入、表示学习和下游训练**。采集由外部真机 SDK 和 MuJoCo replay 仓库完成。训练不会启动 MuJoCo、运行 replay、计算 RBF 或由合力生成仿真 taxel 力。

这是依据 [TactSpace 论文](https://arxiv.org/html/2606.18959) 编写的独立迁移实现，不是作者官方代码，也未复现论文报告的实验指标。

## 当前输入

| 分支 | 局部输入 | 全局输入 |
|---|---|---|
| 真机 | ID3 每个 taxel 的三轴力 `[T,N,3]` | ID3 三轴合力 `[T,3]` |
| 仿真 | ID3 每个 taxel 的接触深度 `[T,N]` | MuJoCo 求解器输出的 ID3 三轴接触合力 `[T,3]` |

默认 `N=120`；位置、法向、taxel 顺序必须与实物标定对应。三轴力采用统一的 ID3 局部坐标系，单位 N；深度采用 m。全局合力是一个独立 token，不复制为每个 taxel 的读数。

**RBF 的边界：**上游若额外导出了 RBF taxel 力，不会因此让整份数据失效。只要选用的 `module_force_sensor_N` 是 RBF 分配前直接累加 MuJoCo 接触力的结果，就可以重新导出为本仓库的 `sim_force_N`。不读取 RBF taxel 力，也不由它们反推深度。文件缺少逐 taxel 深度时，需要上游补齐该字段。

## 目录与阅读顺序

| 文档 | 主要回答的问题 |
|---|---|
| [采集清单与交付要求](docs/collection_plan.md) | 真机和仿真分别采什么、怎样配对、怎样验收？ |
| [当前复现架构](docs/architecture.md) | 输入如何进入两个 ViT、如何重建与对齐、哪些已实现？ |
| [离线数据格式 v2](docs/data_contract.md) | 采集后如何组织 NPZ、元数据和 manifest？ |
| [验证记录](docs/validation.md) | 已做哪些测试、当前结果有哪些限制？ |

```text
TacDex13/
├─ README.md                    项目入口与运行命令
├─ docs/
│  ├─ collection_plan.md         真机／仿真采集清单、分阶段数据和验收要求
│  ├─ architecture.md            论文对应关系、输入与损失设计
│  ├─ data_contract.md           采集后数据格式、单位、配对和划分
│  └─ validation.md              当前验证结果及限制
├─ configs/                     正式配置和短训练配置
├─ src/tacdex/
│  ├─ data.py                   校验、因果窗口、训练集归一化
│  ├─ adapters.py               已解码 SDK 数据的 ID3 提取、离线时间匹配
│  ├─ models.py                 两个 ViT、共享解码器、冻结后的任务头
│  ├─ losses.py                 对比对齐、四条自／交叉重建路径
│  ├─ training.py               两阶段训练、模型保存和测试
│  ├─ cli.py                    命令入口
│  ├─ synthetic.py              明确标注的合成软件测试数据
│  └─ simulation.py             早期可选射线实验工具，离线训练不调用
├─ tests/                       数据、模型与可选仿真工具测试
├─ data/                        本地数据，不进入版本管理
├─ runs/                        权重与指标，不进入版本管理
└─ references/                  外部参考材料，不作为训练依赖
```

采集人员从 [采集清单](docs/collection_plan.md) 开始，模型开发从 [架构](docs/architecture.md) 开始；最终都按 [数据契约](docs/data_contract.md) 对接。下游默认任务为**接触几何／深度图重建**。

## 安装与运行

选择已安装合适 CUDA 版 PyTorch 的 Python 环境。离线训练只要求 PyTorch 和 NumPy：

```powershell
python -m pip install -e ".[dev]"
```

本机已有项目环境，可将下列 `python` 替换为 `.\.venv\Scripts\python.exe`。这个环境继承本机 `lerobot` 环境的 PyTorch，不应拷贝到其他机器。

软件验证用的合成数据与正式采集数据严格区分：

```powershell
python -m tacdex synthetic --out data/demo_v2
python -m tacdex validate --manifest data/demo_v2/manifest.json
python -m tacdex pretrain --manifest data/demo_v2/manifest.json --config configs/smoke_pretrain.json --out runs/demo_v2/alignment --device cuda
python -m tacdex probe --manifest data/demo_v2/manifest.json --pretrained runs/demo_v2/alignment/best.pt --config configs/smoke_probe.json --out runs/demo_v2/probe --device cuda
python -m tacdex evaluate --manifest data/demo_v2/manifest.json --checkpoint runs/demo_v2/probe/best.pt --split test_id --device cuda --out runs/demo_v2/evaluation.json
python -m pytest tests/test_core.py -q
```

正式数据使用相同命令，换成实际 manifest，以及 `configs/id3_pretrain.json`、`configs/depth_probe.json`。输出目录必须是新目录，防止覆盖已有实验。

两阶段必须使用同一个经过校验的 manifest；第二阶段只将仿真深度和仿真合力送入冻结编码器。验证集也只使用仿真输入选下游模型，最终测试时切换为真机分支。

## 兼容性与当前状态

- 当前数据与模型格式为 **v2**，新增必需的 `sim_force_N` 和仿真合力 token／重建头。v1 权重不兼容，需重新预训练；v1 数据可从已保存的原始合力字段重新导出，不能补零迁移。
- SDK 的专有原始字段尚未完成登录克隆后的核对；`extract_id3` 接受已解码数组，不能视作已验证的 ROS2 原始消息解析器。
- MuJoCo 不属于离线训练的必需依赖。早期射线辅助工具需要可选依赖 `.[sim]`，它不改变外部 replay 工程。
- 合成数据用于检查程序、梯度和训练流程，不能证明真机 sim-to-real 有效。
