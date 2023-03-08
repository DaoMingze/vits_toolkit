# Vits Toolkit

[CN](README.md) | [EN](README_en.md) | [CjangCjengh](README_origin.md)

<!--TOC-->

## 部署

```bash
git clone https://github.com/DaoMingze/vits_toolkit
```

- Nvidia GPU
  - CUDA
- Python >=3.7
- torch

## 训练

训练三要素：音频、标注文本、机器可读音素

机器可读音素由标注文本机器转换而来，通过时间轴与音频相关联。

### 训练前：数据集与标注

广义上的数据集包括音频数据集和标注数据集。

样本量与预训练模型：音域、还原度、健壮性

注：未经证实的消息认为vits的多人模型最好不要超过5人，否则单人语音的质量将存在低于更少发音人的质量。而vits多人语音训练的主要特点就是通过多个说话人覆盖音域。即a.当单人样本量足够多的时候，应该训练单人或少人模型。b.当单人样本量少而多人样本量多时，为了支持语音生成的自然率，应当训练多人模型。c.当使用一个音域足够丰富的预训练模型再次训练时，最好只新加入1~2个角色。

#### 音频处理

音频获取

1. 从视频中提取音频
2. 原生音频
3. 简陋录音

#### 对轴标注

如[whisper-vits-japanese](https://github.com/AlexandaJerry/whisper-vits-japanese)所示，我们可以通过ASR（Automatic Speech Recognition, 自动语音识别）来实现根据音频直接生成字幕（时间轴+标注），再通过ffmpeg分割音频，直接生成数据集。

就目前而言，[whisper](https://github.com/openai/whisper)的中文语音识别功能相对孱弱，语音区段划分也有待加强。

- 语音区段识别（打轴）方面，使用[Auditok](https://github.com/amsehili/auditok)，或是集成相关功能的[autosub](https://github.com/BingLingGroup/autosub)。
- 语音切片方面，显然ffmpeg是最佳选择
- 语音识别方面，使用whisper

标注的人工检查与改进

#### 标注文本处理

即`text.cleaners`，将标注文本清理为音素（如拼音、注音、IPA音标、罗马音等）

### 训练时：参数监控

### 训练后：验证

## 模型转换

pt模型的弱兼容性

## 推理

### 文字转语音

### 歌曲

## 精细调整（finetune）

## 策划