# ArmBasic 项目说明

ArmBasic 是一个包含语音交互与人脸识别的多模块项目。核心工作流为“语音输入 → 大模型回答 → 语音播报”，并在需要时调用摄像头做人脸/物体识别，辅助回答。

## 模块概览

- AISpeechInteraction：语音助手模块，负责麦克风录音、语音识别、调用大模型（通义千问）、语音合成与播放，以及必要的视觉识别触发
  - 入口：[speech_ai.py](ArmBasic/AISpeechInteraction/speech_ai.py)
  - 文档：[README.md](ArmBasic/AISpeechInteraction/README.md)
- FaceRecognitionModule：人脸识别模块，负责从摄像头抓取画面并识别人脸，供语音助手作为上下文使用
  - 入口：[run_face_recognition.py](ArmBasic/FaceRecognitionModule/run_face_recognition.py)
  - 文档：[README.md](ArmBasic/FaceRecognitionModule/README.md)

## 环境与依赖

### 系统要求
- macOS (Apple Silicon 优化可用；其他平台需自行替换播放/识别后端)

### Python 依赖
- 语音助手：参见 [AISpeechInteraction/requirements.txt](ArmBasic/AISpeechInteraction/requirements.txt)
- 人脸识别：参见 [FaceRecognitionModule/requirements.txt](ArmBasic/FaceRecognitionModule/requirements.txt)

### 额外系统依赖（macOS）
- PortAudio（PyAudio 依赖）
- flac（语音识别编码工具）

一键安装（推荐）：
```bash
cd AISpeechInteraction
chmod +x install_mac.sh
./install_mac.sh
```
或手动：
```bash
brew install portaudio flac
pip install -r AISpeechInteraction/requirements.txt
pip install -r FaceRecognitionModule/requirements.txt
```

## 配置

在通义千问控制台创建 API Key，并在语音助手模块下创建配置：
```bash
cp AISpeechInteraction/config_example.env AISpeechInteraction/.env
```
编辑 `.env`：
```
DASHSCOPE_API_KEY=sk-你的key
```

人脸识别样本库目录：将你的脸部照片放入  
[FaceRecognitionModule/known_faces](ArmBasic/FaceRecognitionModule/known_faces)

## 运行

### 语音助手
```bash
python AISpeechInteraction/speech_ai.py
```
- 唤醒词：小笨（支持“小本”“笨笨”等同音模糊）
- 在说「我是谁」「看看我是谁」等时，会自动触发摄像头拍照并进行人脸识别，结果作为上下文传给大模型

### 文字模式（无需麦克风）
```bash
python AISpeechInteraction/speech_ai.py --text
```

## 主要能力与特性

- 语音识别
  - 优先使用 Whisper tiny（支持 Apple Silicon 的 MPS 加速，速度更快）
  - 回退为 Google 语音识别（国内网络下可改造为本地/国产 ASR）
  - 识别参数优化：缩短片段、降低停顿阈值，减少整体延迟  
    参考实现：[listen](ArmBasic/AISpeechInteraction/speech_ai.py#L579-L633)
- 打断与唤醒
  - 播报为非阻塞，播放中仍持续监听唤醒词（例如“小笨”）  
    参考实现：[run](ArmBasic/AISpeechInteraction/speech_ai.py#L635-L705)
  - 打断同时停止 TTS 与大模型流式输出（不只停播，更会停生成）  
    参考实现：[停止逻辑 + LLM 停止事件](ArmBasic/AISpeechInteraction/speech_ai.py#L465-L492)
- 大模型（Qwen）
  - 文本流式输出，首字响应更快  
    参考实现：[chat_with_llm(stream=True)](ArmBasic/AISpeechInteraction/speech_ai.py#L157-L226)
- 语音合成与播放
  - edge-tts 合成，分句流式播报，降低首字延迟  
    参考实现：[AudioPlayer.play_stream](ArmBasic/AISpeechInteraction/speech_ai.py#L244-L352)
  - 静态提示音缓存（“我在”“请吩咐”等），唤醒响应毫秒级  
    参考实现：[静态音频缓存](ArmBasic/AISpeechInteraction/speech_ai.py#L531-L560)
- 摄像头与人脸识别
  - 触发词识别 → 拍照 → 人脸识别 → 注入上下文  
    参考实现：[视觉触发与拍照](ArmBasic/AISpeechInteraction/speech_ai.py#L714-L755)

## 性能与稳定性优化摘要

- 固定能量阈值，避免扬声器播报导致的阈值漂移
- 缩短识别窗口与停顿判定，减少 3~5 秒等待
- MPS 加速 Whisper（Apple Silicon），显著提升推理速度
- 流式 TTS 分句播报，首字响应快
- 打断时同时停止 LLM 与 TTS，并清理临时音频文件，避免缓存堆积

## 常见问题

- PyAudio 安装报错  
  先安装 `portaudio`，再 `pip install pyaudio`。可使用 `AISpeechInteraction/install_mac.sh` 一键安装。
- Google 识别不可用  
  可仅使用 Whisper；或替换为国产 ASR（如 FunASR），需要改动 `listen` 实现。
- 播放器在非 macOS  
  当前使用 `afplay`，其他系统需要替换为兼容的播放命令或库。

## 目录结构（简要）

```
ArmBasic/
├─ AISpeechInteraction/
│  ├─ speech_ai.py
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ install_mac.sh
│  └─ static_audio/
├─ FaceRecognitionModule/
│  ├─ run_face_recognition.py
│  ├─ requirements.txt
│  └─ known_faces/
└─ .gitignore
```

## 许可

本项目用于学习与实验目的，涉及云 API 的使用请遵循各服务条款与额度限制。
