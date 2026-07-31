# Spec: add-fallback-handling

## Requirement: 依赖缺失兜底

### Scenario: 可选依赖缺失
- **Given** 某依赖未安装
- **When** 启动助手
- **Then** 该能力降级禁用，其余功能正常
- **And** 启动日志汇总缺失项为一条 WARNING

## Requirement: 麦克风不可用

### Scenario: PyAudio 初始化失败
- **Given** 无麦克风或 PortAudio 缺失
- **When** 启动语音模式
- **Then** 自动切换文字模式
- **And** 播报"麦克风不可用，已切换文字模式"

## Requirement: ASR 兜底

### Scenario: Whisper 失败回退 Google
- **Given** Whisper 模型缺失或推理出错
- **When** 调用 ASR
- **Then** 回退 Google 识别
- **And** 两者均失败返回空文本并播报"没听清，请再说一次"

## Requirement: 大模型兜底

### Scenario: 网络超时重试
- **Given** LLM 调用超时或网络错
- **When** 触发 chat_with_llm
- **Then** 重试 3 次，退避 1/2/4 秒
- **And** 重试耗尽播报"网络异常，请稍后再试"，主循环继续

### Scenario: API 429 不重试
- **Given** 大模型返回 429
- **When** 触发调用
- **Then** 立即降级，不重试
- **And** 播报限流提示，主循环继续

## Requirement: TTS/播放兜底

### Scenario: edge-tts 或 afplay 失败
- **Given** TTS 网络错或播放器缺失
- **When** 需要播报
- **Then** 跳过播报，回复文本打印到终端
- **And** 主流程不中断

## Requirement: 摄像头与人脸库兜底

### Scenario: 摄像头不可用
- **Given** VideoCapture 打不开
- **When** 触发视觉识别
- **Then** 跳过视觉，仅用语音上下文
- **And** 播报"摄像头不可用"

### Scenario: 人脸库空
- **Given** known_faces 为空或加载失败
- **When** 调用人脸识别
- **Then** 返回空结果
- **And** 播报"未识别到已知人脸"

## Requirement: 配置缺失兜底

### Scenario: API Key 未配置
- **Given** .env 缺失或无 DASHSCOPE_API_KEY
- **When** 启动
- **Then** 禁用 LLM 能力
- **And** 播报"未配置API Key"

## Requirement: 日志

### Scenario: 异常记录
- **Given** 任一兜底触发
- **When** 发生降级
- **Then** 写入 logs/armbasic.log（WARNING+）与控制台
- **And** 不吞掉真问题（栈信息记录）
