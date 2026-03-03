# SimpleSV

## 1. 简介

SimpleSV是一款基于ECAPA-TDNN的说话人识别模型。采用CN-Celeb1数据集进行训练，使用ECAPA-TDNN作为基础模型，结合AAM-Softmax损失函数进行优化。验证阶段使用余弦相似度进行匹配。以当前参数进行训练，在测试集上达到了约71.53%的准确率。

本项目的代码结构如下：

```text
SimpleSV
├─ README.md
├─ requirements.txt
└─ src
   ├─ main.py               # 主程序入口
   └─ network
      ├─ classifier.py      # AAM分类器定义
      ├─ config.py          # 模型配置定义
      ├─ dataset.py         # 数据集定义
      ├─ model.py           # SV模型与训练器定义
      ├─ tdnn.py            # ECAPA-TDNN网络定义
      └─ __init__.py
```

## 2. 数据集

本说话人识别模型采用的数据集是CN-Celeb1。该数据集包含了大量的中文说话人语音样本，适用于中文说话人识别任务。该数据集主要分为训练数据集和测试数据集两部分，每个数据集中的语音样本均为16kHz采样率的单声道音频文件，平均每个说话人包含约126.91个语音样本。本项目将原生学习数据集（共108188条数据）按照99:1的比例划分为训练集（共107106条数据）和验证集（共1082条数据）。测试数据集主要包含声纹注册数据（共196条数据）和声纹验证数据（共17777条数据）。

数据集的下载链接：[CN-Celeb1](https://openslr.magicdatatech.com/82/)。请下载并解压文件cn-celeb_v2.tar.gz，并将`main.py`中常量`DATASET_ROOT_DIR`设置为解压后的CN-Celeb_flac目录的路径。

## 3. 模型架构

本项目采用的模型架构是EPACA-TDNN，模型主要参数与训练策略均参考自论文[ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)。模型的输入为MFCC特征，输出为说话人嵌入向量。训练过程中使用AAM-Softmax损失函数进行优化。优化器采用Adam，采用triangular2学习率调度策略，并附加L2正则化抑制过拟合。

## 4. 快速开始

### 4.1. 环境配置

该项目需要一个基于GPU的Pytorch环境。请确保已安装PyTorch和相关依赖库。可以使用以下命令安装所需的Python包：

```bash
pip install -r requirements.txt
```

### 4.2. 数据准备与路径配置

请下载并解压CN-Celeb1数据集，并将`main.py`中常量`DATASET_ROOT_DIR`设置为解压后的CN-Celeb_flac目录的路径。

同时，需要将常量`TDNN_NETWORK_PATH`设置为训练产出模型存放位置/测试模型位置，`LOG_DIR`设置为训练日志存放目录，`CKPT_DIR`设置为训练检查点存放目录。

以上路径建议使用绝对路径，以避免路径解析错误。

### 4.3. 模型配置

`config.py`中提供了模型的默认配置参数。如果需要使用自己的模型参数，不建议直接在`config.py`中修改默认配置，而是建议在`main.py`中配置实例化处进行修改。

### 4.4. 调用指令

运行`main.py`即可启动程序。初始化后，模型默认提供7种指令：

- Show dataset info: 显示数据集信息
- Show model info: 显示模型信息
- Train model (first run): 训练模型（首次运行）。该命令会清空所有日志、检查点文件，并重新开始训练，需要二次确认。
- Train model (from checkpoint): 训练模型（继续）。该命令会从上次训练中断的地方继续训练。请确保运行该命令前已经生成了至少一个检查点文件。
- Test model: 测试模型。该命令会使用最终模型在测试集上进行准确率测试，并输出测试结果。请确保运行该命令前已经生成了最终模型文件。
- Run TensorBoard: 运行TensorBoard。该命令会启动TensorBoard服务器，用户可以通过浏览器访问TensorBoard界面查看训练日志。请确保运行该命令前已经生成了训练日志文件。
- Exit: 退出程序

输入命令对应的数字即可执行相应的操作。

## 5. 未来计划

### 5.1. 模型优化

通过反复实验和调参，进一步优化模型性能，提升说话人识别的准确率。

### 5.2. 构建声纹管理系统

开发UI界面，构建一个完整的声纹管理系统，支持用户注册、验证和管理声纹数据。

### 5.3. 向量数据库支持

集成向量数据库（如FAISS），实现高效的说话人嵌入向量存储和检索，提升系统的响应速度和扩展性。

### 5.4. 识别算法改进

当前模型识别算法采用的是余弦相似度，未来计划引入更先进的识别算法，如PLDA（Probabilistic Linear Discriminant Analysis）等，以进一步提升识别性能。
