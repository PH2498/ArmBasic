# Design: add-fallback-handling

## Tech Stack（实测）

Python，两模块：`AISpeechInteraction/speech_ai.py`（812行，Whisper/Google ASR→Qwen→edge-tts→afplay）、`FaceRecognitionModule/run_face_recognition.py`（338行，cv2+face_recognition）。macOS Apple Silicon。

## 失败点与兜底

| 失败点 | 异常 | 兜底 | 用户提示 |
| --- | --- | --- | --- |
| 依赖缺失 | ImportError | 已有 `HAS_*` 标志；缺失时降级该能力 | 启动时日志列出缺失项 |
| 麦克风/PyAudio | OSError/无设备 | 跳过语音模式，提示用 `--text` | "麦克风不可用，已切换文字模式" |
| ASR | Whisper模型缺/Google网络 | Whisper 失败回退 Google，均失败返回空文本 | "没听清，请再说一次" |
| 大模型 | 超时/key无效/429 | 3次重试(1/2/4s)，429不重试；失败播报 | "网络异常，请稍后再试" |
| TTS | edge-tts网络/afplay缺失 | 跳过播报，文本打印到终端 | 终端输出回复文本 |
| 摄像头 | VideoCapture失败 | 跳过视觉，仅语音上下文 | "摄像头不可用" |
| 人脸库空/加载失败 | 空目录/IOError | 返回"未识别到已知人脸" | "未识别到已知人脸" |
| 配置缺API Key | .env缺失 | 启动即检查，缺失时禁用LLM并提示 | "未配置API Key" |
| 临时音频清理 | OSError | 忽略，仅日志 | 无 |

## fallback.py 接口

```python
class FallbackError(Exception):
    """兜底后仍无法恢复时抛出，由主循环捕获继续。"""

def with_retry(fn, retries=3, delays=(1,2,4), on_network=True): ...
def safe_call(fn, fallback=None, log_msg="", user_hint=""): ...
def log_warning(msg, exc=None): ...   # 控制台 + logs/armbasic.log
def user_say(text): ...               # 调 TTS 或终端打印
```

## 接入点（最小改动）

- `speech_ai.py#listen`：包 `safe_call`，ASR 失败降级。
- `speech_ai.py#chat_with_llm`：包 `with_retry`，超时/429 降级。
- `speech_ai.py#AudioPlayer.play_stream`：包 `safe_call`，播放失败终端打印。
- `speech_ai.py` 视觉触发段：摄像头失败跳过。
- `run_face_recognition.py` 主循环：`VideoCapture` 打不开 return；人脸库空 return `[]`。

## 日志

`logging` 模块，`WARNING` 及以上写 `logs/armbasic.log`（自动建目录），同时控制台。启动时汇总缺失依赖为一条 WARNING。

## 不确定项

无（已按默认值确定，见 proposal.md Defaults）。
