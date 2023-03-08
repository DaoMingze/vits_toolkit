# VITS工具集

[origin](README.md) | **[ZH](README_zh.md)** | [EN](README_en.md)

## 修改

### 原则

1. 基于`black`实现的python代码规范
2. 为i5-9600KF + RTX 2070 (clevo) , Windows 11 + python 3.10 + pytorch 1.13.1+cu117上运行适配
3. python使用pip进行包管理，不使用conda。
4. 代码编辑使用vscode。
5. 对代码进行最小改动。

### 具体项

#### 文档

1. 论文文本翻译
2. 常见问题解决

#### 预处理

对cleaner文件的方言部分进行了使用时加载，规避了方言转换字典加载失败的问题。

#### 训练

由于在Windows上训练，将`train.py`的67行后端（backend）从nccl改为gloo。

根据硬件性能和报错建议，将`train.py`的80行，num_workers降为6。

小数据的对等训练（训练量=校验量）

提供MB-iSTFT-VITS的训练

```cmd
完全相同
attentions.py
commons.py
modules.py
preprocess.py
transforms.py
修改添加
losses.py	#导入iSTFT中关于subband_stft_loss(h, y_mb, y_hat_mb)的函数定义、对sftf_loss的引用
utils.py	#日志输出等级统一为debug，模型生成路径改为"../model"，将自动删除3000步以上过时模型的命令从Linux系统的`rm`改为Windows cmd的`del`
stft.py	#来自iSTFT，增加auto关于cartesian_inverse(self, real, imag)的定义
model.py	#加入iSTFT的iSTFT、MB、MS三个生成类，修改增加合成类参数（gen_isftf_n_fft、gen_isftf_hop_size、ms/mb/subbands/isftf_vits，6个）
新增
pqmf.py	#来自iSTFT
stft_loss.py	#来自iSTFT
train_istft.py		#来自iSTFT
train.py`#来自VITS
train_ms.py	#来自VITS
冲突
三个都不一样，后缀区分
mel_processing.py	#iSTFT与auto冲突，后缀区分
data_utils.py	#iSTFT与auto冲突，后缀区分
```

#### 推理/合成

## 安装：获取、运行环境及依赖

### 获取

克隆或下载本仓库→`git clone https://github.com/DaoMingze/vits_toolkit`

### 运行环境

1. 安装NVIDIA CUDA（本仓库使用的是cuda 11.7）和cudnn工具包（本仓库使用的是cudnn 8.6.0）
2. 安装[Python](https://www.python.org)（本仓库使用的是python 3.10）
3. 更新pip→`pip install -U pip`
4. 使用pip安装与之对应的[torch](https://www.pytorch.org)cuda版本（本仓库使用的是 1.13.1+cu117）
5. 切换到本仓库所在路径/目录
6. 基于个人情况，选择是否构建python虚拟环境、虚拟环境是否包含系统第三方库（include-system-site-packages）
7. 使用pip安装常规依赖→`pip install -r requirements.txt`
8. **注意**：安装单调对齐`monotonic_align`↓

```cmd
cd monotonic_align #进入单调对齐文件夹
md monotonic_align #创建目录
python setup.py build_ext --inplace # 在此安装
cd .. #返回上级目录
```

运行环境的一些说明：

- python >=3.6
- torch >=1.6.0

报错较少，符合原项目环境。

### 依赖

#### espeak

英语模型的训练与使用，可能需要espeak，该软件原版已停止更新，现可用[espeak-ng](https://github.com/espeak-ng/espeak-ng)

由于`phonemizer`库只接受espeak，因此Windows环境需要在系统环境变量中加入：

```cmd
PHONEMIZER_ESPEAK_PATH: "C:\Program Files\eSpeak NG"
PHONEMIZER_ESPEAK_LIBRARY: "C:\Program Files\eSpeak NG\libespeak-ng.dll"
```

其中，`C:\Program Files\eSpeak NG`是espeak-ng的默认安装路径，请根据具体情况修改。

对于Linux用户，可以使用`apt-get install espeak`完成。

#### audonnx库

如果使用W2V2作为情感模块，则应安装audonnx库以支持对应预训练模型使用。

```python
pip install audonnx
```

#### 汉语方言

cjang在文本预处理中引用了方言词汇转换，但没有给出方言转换的配置。目前opencc本身也没有支持方言，因此调用会报错。

如有方言训练的需求，可以对cjang的另一个方言词汇库做处理后导入opencc，也可以利用sky项目中做好的ocd2、json文件。

## 理解文件作用

文本清洗（text cleaner），各原仓库基于[keithito's tacotron](https://github.com/keithito/tacotron)，仅支持英文。根据[CjangCjengh的vits分支](https://github.com/CjangCjengh/vits)，加入了日语、汉语、韩语、梵语和泰语的支持

## 训练

vits语音模型的训练

### 准备数据集

需要准备标注了语音内容及其波形文件映射关系的文本文件，具体要求如下

#### 音频格式

> 16-bit PCM WAV，22050 Hz，单声道
>
> 16比特的脉冲编码调制(Pulse Code Modulation,PCM)的波形数字音频封装格式（WAV），采样率22050Hz，单声道

#### 文本标注格式

> `path/speaker/filename.wav|speaker_id|context`
>
> 音频文件路径/说话人/波形文件名|说话人识别号|语音内容
>
> 当为单人时，说话人（speaker）及其识别号（speaker_id）可省略，识别号从0开始编号。

例如

```text
a speaker ##单人
wavs/filename.wav|context
mult speakers ##多人
wavs/A/001.wav|0|context
wavs/A/002.wav|0|context
wavs/B/001.wav|1|context
wavs/B/002.wav|1|context
```

#### 配置

注意：默认情况下

> filelists，文本标注文件夹
>
> wavs，音频文件夹

- model_filename，生成模型文件名
- batch_size，批处理数量
- epochs，训练次数
- train_files: 训练文本路径，即标注
- validation_files: 检验文本路径，即标注

训练文本与检验文本，标注的内容应该不同。

### 开始训练

标注文本预处理

```python
python preprocess.py --text_cleaners chinese_cleaners --filelists filelists/xi_train.txt filelists/xi_val.txt
python preprocess.py --text_cleaners japanese_cleaners --filelists filelists/isla_train.txt filelists/isla_val.txt

```

- `--text_index`，1为单人，2为多人，默认为1
- `--text_cleaners`，处理格式
- `--filelists`，标注文本及其处理后文本路径
  1. 训练文本
  2. 校验文本

```python
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
python train.py -c configs/xi.json -m xi_base
python train_istft.py -c configs/xi_istft.json -m xi_istft
python train_auto.py -c configs/xi_auto.json -m xi_auto
```

- `-c`，配置文件路径
- `-m`，模型名称

### 训练性能

单位：步/分钟；x是暂未开始训练

|epochs|VITS|iSTFT|AutoVocoder|
|---|---|---|---|
|100|3.57|2.78|x|
|200|2.56[^1]|2.74|x|
|300|2.4|x|x|

[^1]: 由于VITS可以在已有模型上继续训练，日志显示是从72到200，即实际128。

## 模型

vits的三类应用模型，fork from [CjangCjengh's TTSModels](https://github.com/CjangCjengh/TTSModels)

`git clone https://github.com/CjangCjengh/TTSModels.git model`

### VITS原版

[VITS](https://github.com/jaywalnut310/vits)

### HuBERT-VITS

HuBERT内容编码器，是[一个为提高语音转换的离散和浊化的语音单元对照](https://github.com/bshall/soft-vc)的三个组件之一（其余是声学模型和声码器）。

<details>

该语音转换系统的架构（来自原文图注）

1. 离散内容编码器将音频特征聚类以产生一系列离散语音单元。
2. 软内容编码器被训练来预测离散单元。
3. 声学模型将离散/软语音单元转换为目标频谱图。
4. 声码器将频谱图转换为音频波形。

> 语音转换的目标是将源语音转换为目标语音，同时保持内容不变。 在本文中，我们专注于语音转换的自监督表征学习。 具体来说，我们将离散和软语音单元作为输入特征进行比较。 我们发现离散表示有效地删除了说话人的信息，但丢弃了一些语言内容——导致发音错误。作为解决方案，我们提出通过预测离散单元的分布来学习软语音单元。 通过对不确定性建模，软单元捕获更多内容信息，提高转换后语音的清晰度和自然度。——[soft-vc](https://github.com/bshall/soft-vc)

关于[HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md)

[论文](https://arxiv.org/abs/2111.02392)

</details>

HuBERT-VITS仅使用其中的HuBERT-Soft模型支持语音转换，[在此获取](https://github.com/bshall/hubert/releases)

### W2V2-VITS

W2V2是基于维度的语音情感识别模块Wav2vec2.0的缩写，该模块以CC BY-NC-SA 4.0开放使用。这里提供的是在[MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) (v1.7)上预训练的[wav2vec2-large-robust](https://huggingface.co/facebook/wav2vec2-large-robust)，它在精调之前从24transformer层修剪精调至12层。[w2v2-how-to](https://github.com/audeering/w2v2-how-to)提供torch框架下的onnx格式文件，[在此获取](https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip?download=1)[^2]。

[^2]: 在中国大陆可能需要借助代理工具访问

**注意**：在使用该模块前，应安装audonnx（python库）以支持onnx模型→`pip install audonnx`

## 合成

```python
python ..\MoeGoe\MoeGoe.py
```

## 参考

- [VITS](https://github.com/jaywalnut310/vits)：具有用于端到端文本到语音的对抗性学习的条件变分自动编码器，原理详见[论文](https://arxiv.org/abs/2106.06103)。
  - [CjangCjengh的vits](https://github.com/CjangCjengh/vits)，支持日汉韩梵泰的改进，本仓库的fork对象。
- [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)，具有多波段生成和逆短时傅里叶变换的轻量级高保真端到端文本到语音，原理详见[论文](https://arxiv.org/abs/2210.15975)。
  - [多语言版](https://github.com/vitzhdanov/MB-iSTFT-VITS-multilingual)，主要是文本语言处理的改进。
- [MB-iSTFT-VITS-with-AutoVocoder](https://github.com/hcy71o/MB-iSTFT-VITS-with-AutoVocoder)，将基于 iSTFTNet 的解码器替换为基于 AutoVocoder 的解码器。

- [VITS](https://github.com/jaywalnut310/vits)
- <https://zhuanlan.zhihu.com/p/419883319>
- config格式注解，参考<https://www.bilibili.com/read/cv18478187>
