# MSRDA-GSCA

基于 ASRT 改进的中文语音识别研究项目，采用 TensorFlow/Keras 实现多尺度残差卷积、通道与空间注意力融合，以及 CTC 声学建模。项目提供训练、评估、单音频识别和 HTTP/gRPC 服务入口。

本仓库基于 [ASRT Speech Recognition](https://github.com/nl8590687/ASRT_SpeechRecognition) 修改，保留原作者版权声明和 [GPL-3.0 许可证](LICENSE)。原始中文说明保存在 [README_ASRT.md](README_ASRT.md)，原始英文说明见 [README_EN.md](README_EN.md)。这些上游说明中的模型、下载地址及实验结果不代表本项目的验证结果。

## 模型结构

核心实现位于 `model_zoo/speech_model/keras_backend.py` 的 `SpeechModel251BN` 类，内部模型名为 `SpeechModel251bn_multiscale`。

- **多尺度残差分支**：并行使用 1×1、3×3、5×5 和膨胀率为 2 的 3×3 卷积；各分支加入残差连接，拼接后通过 1×1 卷积投影。
- **局部注意力融合**：`HeBing` 将 `TongDao` 通道注意力和 `KongJian` 空间注意力的输出逐元素取最大值。
- **后端联合注意力**：`GCSA` 依次执行通道注意力、通道重排和空间注意力。项目名称使用 GSCA，当前代码类名使用 GCSA。
- **CTC 识别**：默认输入尺寸为 `(1600, 200, 1)`，时间维下采样 8 倍；输出 1428 类，对应拼音字典及 CTC 空白符。

```text
WAV → 频谱特征 / SpecAugment
    → 多尺度残差模块 + HeBing（32 → 64 → 128 通道）
    → 卷积特征提取 → GCSA
    → Reshape → Dense → Softmax → CTC 解码 → 拼音
    → 语言模型 → 中文文本
```

仓库中的 PyTorch 后端为另一套保留实现，不能直接视为上述改进模型的等价复现。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `model_zoo/speech_model/keras_backend.py` | 改进声学模型及保留的基线模型 |
| `train_speech_model.py` | TensorFlow 训练、参数量与 FLOPs 分析 |
| `evaluate_speech_model.py` | 加载权重并评估，默认使用 dev 集 |
| `predict_speech_file.py` | 单个音频文件识别 |
| `speech_model.py` | 训练、解码与评估封装 |
| `speech_features/` | 频谱特征提取与数据增强 |
| `data_loader.py`、`asrt_config.json` | 数据加载与数据集路径配置 |
| `datalist/`、`dict.txt` | 数据索引、拼音标签和字典 |
| `model_language/`、`language_model3.py` | 拼音转中文的语言模型 |
| `asrserver_http.py`、`asrserver_grpc.py` | 服务接口 |
| `client_http.py`、`client_grpc.py` | 调用示例 |
| `RTF.py` | 模型复杂度与推理速度测量脚本 |

## 环境准备

项目保留了实验时的 `requirements.txt`，其中包含 `tensorflow-gpu==2.8.4`、`numpy==1.24.1` 等固定版本。当前依赖组合尚未完成干净环境安装验证，请根据目标平台及 TensorFlow 的 Python/CUDA 支持情况建立独立环境；遇到版本冲突时需要调整依赖。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 则使用：.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

gRPC 入口额外依赖 `grpcio`；PyTorch 入口额外依赖 `torch`，均未列入当前依赖文件。仓库附带的 Dockerfile 使用另一套历史依赖，尚未验证与当前改进模型兼容。

## 数据准备

音频数据和训练权重需自行准备，不随代码仓库发布。建议使用与当前特征提取器一致的 16 kHz 单声道 WAV 音频。

1. 准备音频、音频路径列表和对应拼音标签。
2. 编辑 `asrt_config.json`，设置 `train`、`dev`、`test` 中各数据集的 `data_path`、`data_list` 和 `label_list`。
3. 删除未准备好的数据集配置。默认配置列出了 THCHS-30、ST-CMDS、Primewords、AISHELL-1、aidatatang 和 MagicData；配置存在不代表本地数据齐全。
4. 确认标签与 `dict.txt` 一致。默认最大标签长度为 64，模型输入最多为 1600 帧。

配置项示例（路径需按实际目录修改）：

```json
{
  "name": "thchs30_train",
  "data_list": "datalist/thchs30/train.wav.lst",
  "data_path": "/data/speech_data",
  "label_list": "datalist/thchs30/train.syllable.txt"
}
```

`download_default_datalist.py` 可用于下载上游默认索引和标签；它不会替代音频数据集下载，运行前请检查脚本中的来源和目标目录。

## 训练

从项目根目录运行：

```bash
python train_speech_model.py
```

当前训练参数直接写在脚本中：

| 参数 | 默认值 |
| --- | --- |
| 优化器 | Adam |
| 学习率 | 0.0005 |
| Epochs | 50 |
| Batch size | 4 |
| 最大标签长度 | 64 |
| 输入尺寸 | 1600 × 200 × 1 |

运行前按设备情况检查 `CUDA_VISIBLE_DEVICES`，并确认 `save_models/` 目录可写。脚本会尝试输出训练图及纯声学推理图的复杂度；FLOPs 取决于输入形状、框架和统计范围，应保留测量条件。

## 评估与推理

评估脚本默认加载 `save_models/SpeechModel251bn_multiscale.model.h5`，使用配置中的 dev 集并输出报告：

```bash
python evaluate_speech_model.py
```

若需评估 test 集，将脚本中的 `DataLoader('dev')` 改为 `DataLoader('test')`。加载权重必须与当前模型结构和字典一致。

单文件识别前，将 `predict_speech_file.py` 中的 `filename.wav` 替换为实际音频路径，然后运行：

```bash
python predict_speech_file.py
```

程序先输出声学模型的拼音结果，再使用 `model_language/` 中的语言模型转换为中文。

## 服务接口

准备好相同结构的权重后，可查看服务参数并启动：

```bash
python asrserver_http.py --help
python asrserver_grpc.py --help
python asrserver_http.py
# 或启动 gRPC 服务
python asrserver_grpc.py
```

调用示例位于 `client_http.py` 和 `client_grpc.py`，使用前需修改音频路径及服务地址。

## 复现状态与限制

- 本仓库提供代码和现有数据索引，不包含训练权重、原始音频、缓存及本地实验报告。
- 本次整理依据静态代码检查，未执行完整环境安装、模型训练、数据集评估或服务端到端测试。
- `RTF.py` 使用脚本内的输入与音频时长假设；其结果不能直接当作真实语音流水线的端到端 RTF。
- 暂不声明识别率、相对提升幅度或跨平台兼容性。报告实验结果时，应注明数据划分、权重、随机种子、硬件和软件环境。

## 致谢与许可

感谢 ASRT 项目及其作者提供中文语音识别基础实现。许可证见 [LICENSE](LICENSE)；源码保留原有版权声明。数据集请遵守各自的授权条款。
