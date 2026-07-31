# Tasks: add-fallback-handling

> 实现阶段执行清单，propose 阶段不勾选完成项。

## fallback 模块

- [ ] 新增 `AISpeechInteraction/fallback.py`：`FallbackError`、`with_retry`、`safe_call`、`log_warning`、`user_say`、日志初始化（建 `logs/` 目录）。
- [ ] 单元测试 `test_fallback.py`：with_retry 重试次数与退避、safe_call 降级返回、429 不重试、日志写入。

## 接入 speech_ai.py

- [ ] `listen`：用 `safe_call` 包裹，ASR 失败返回空文本并 `user_say("没听清，请再说一次")`。
- [ ] `chat_with_llm`：用 `with_retry` 包裹，超时/网络错重试，失败 `user_say("网络异常，请稍后再试")`。
- [ ] `AudioPlayer.play_stream`：用 `safe_call` 包裹，播放失败终端打印文本。
- [ ] 视觉触发段：摄像头打开失败跳过视觉，`user_say("摄像头不可用")`。
- [ ] 启动检查 `.env`/`DASHSCOPE_API_KEY`，缺失禁用 LLM 并 `user_say("未配置API Key")`。
- [ ] 麦克风/PyAudio 初始化失败：自动切 `--text` 并提示。
- [ ] 临时音频清理失败：忽略，`log_warning`。

## 接入 run_face_recognition.py

- [ ] `VideoCapture` 打不开：`log_warning` 并 return。
- [ ] `known_faces` 空或加载失败：return `[]`，`user_say("未识别到已知人脸")`。
- [ ] 字体加载失败：保留现有 `pass`（已兜底）。

## 配置与日志

- [ ] 默认值常量：`ASR_TIMEOUT=15`、`LLM_TIMEOUT=30`、`TTS_TIMEOUT=10`、`RETRIES=3`、`DELAYS=(1,2,4)`、`LOG_FILE=logs/armbasic.log`。
- [ ] 启动汇总缺失依赖为一条 WARNING。

## 验收

- [ ] `python -c "import ast; ast.parse(open('AISpeechInteraction/fallback.py').read())"` 语法通过。
- [ ] `pytest test_fallback.py` 通过。
- [ ] 拔网/缺摄像头场景冒烟：助手不崩溃，给出对应提示。
