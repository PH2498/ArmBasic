# 语音 AI 交互（AISpeechInteraction）

语音输入 → 大模型 → 语音输出，并支持**调用人脸识别**：当你说「我是谁」「看看我是谁」时，会打开摄像头识别你的人脸并让 AI 根据识别结果回答。

## 推荐国内免费/低成本大模型

| 厂商     | 模型       | 说明 |
|----------|------------|------|
| **阿里 通义千问** | qwen-turbo | 本模块默认使用。新用户有免费额度，[控制台](https://dashscope.console.aliyun.com/) 创建 API Key。 |
| 百度 文心 | ERNIE-Speed | 千帆平台有免费额度，需改代码中的 API 调用。 |
| 智谱 GLM | glm-4-flash | 有免费额度，需改代码中的 API 调用。 |

## 安装（含麦克风）

**PyAudio 依赖系统库 PortAudio**，必须先装再 `pip install`，否则会报错 `portaudio.h file not found`。

**macOS 一键安装（推荐）：**

```bash
cd AISpeechInteraction
chmod +x install_mac.sh
./install_mac.sh
```

或手动执行：

```bash
brew install portaudio flac
pip install -r requirements.txt
```

- **portaudio**：PyAudio 依赖  
- **flac**：语音识别转 FLAC 用；Apple Silicon (M1/M2/M3) 必须装，否则会报 `Bad CPU type in executable`  

未安装 Homebrew 请先安装：<https://brew.sh>

## 配置

1. 复制 `config_example.env` 为 `.env`。
2. 在 [阿里云 DashScope](https://dashscope.console.aliyun.com/) 创建 API Key，填入 `.env`：
   ```
   DASHSCOPE_API_KEY=sk-你的key
   ```

人脸数据使用项目内 **FaceRecognitionModule/known_faces**：请在该目录下放入你的照片（如 `张三.jpg`），问「我是谁」时才会被识别并回答姓名。

## 运行

- **语音模式**（需麦克风）：
  ```bash
  python speech_ai.py
  ```
- **文字模式**（不用麦克风，便于测试）：
  ```bash
  python speech_ai.py --text
  ```

说或输入「我是谁」「看看我是谁」等时，程序会调用摄像头做人脸识别，并把「当前识别到的人脸：XXX」注入给大模型，由 AI 用语音回答。

## 流程简述

1. 麦克风/文字得到用户输入。
2. 若检测到「我是谁」类问题 → 调用 `FaceRecognitionModule` 的 `recognize_faces_from_camera()` 拍一帧并识别。
3. 将识别结果作为上下文与用户问题一起发给通义千问。
4. 使用 edge-tts 将回复转为语音播放。

若在国内无法使用 Google 语音识别，可用 `--text` 用文字输入测试；或自行替换 `listen_microphone()` 为百度/阿里 ASR。
