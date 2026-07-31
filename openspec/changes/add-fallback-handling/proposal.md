# Proposal: add-fallback-handling

## What

为 ArmBasic（语音助手 + 人脸识别）新增统一的异常兜底方案：对依赖缺失、麦克风、ASR、大模型、TTS、摄像头、人脸库、配置等失败点提供降级而非崩溃的处理，保证单点故障不中断主循环。

## Why

现状仅 `try/except ImportError` 可选导入与字体 `pass` 兜底。README 已列 PyAudio/Google/afplay 等已知失败点，但运行中任一环节抛异常会中断整个助手循环。需要统一兜底，使设备在弱网/缺依赖/硬件不可用时仍可用核心能力并给出语音提示。

## Scope

### In
- 新增 `AISpeechInteraction/fallback.py`：统一异常类型、重试装饰器、降级决策、日志。
- 改造 `speech_ai.py`：listen/chat_with_llm/AudioPlayer/视觉触发接入兜底。
- 改造 `run_face_recognition.py`：摄像头打开/人脸库加载兜底。
- 新增配置默认值（超时/重试/日志路径）。

### Out
- 不替换底层引擎（仍用 Whisper/Qwen/edge-tts/face_recognition）。
- 不新增 ASR/TTS 引擎国产化（README 已述为未来改造）。
- 不做跨进程守护/看门狗。

## Risk

| 风险 | 缓解 |
| --- | --- |
| 兜底吞异常掩盖真问题 | 日志记录 WARNING+，降级时 TTS 提示用户 |
| 重试放大限流 | 仅网络类重试，API 429 立即降级不重试 |

## Rollback

纯新增 `fallback.py` + 调用点 try/except 包裹；回滚删除该文件并还原调用点即可。

## Defaults（确认项已定）

- 兜底策略：降级不崩溃
- 重试：网络类 3 次，指数退避 1/2/4 秒
- 超时：ASR 15s / LLM 30s / TTS 10s
- 日志：控制台 + `logs/armbasic.log`，WARNING 及以上
- 用户提示：TTS 播报友好语（如"网络异常，请稍后再试"）
- 摄像头不可用：跳过视觉，仅语音
- 人脸库空：返回"未识别到已知人脸"
