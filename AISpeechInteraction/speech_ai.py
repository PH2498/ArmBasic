"""
语音 AI 交互：语音输入 -> 大模型 -> 语音输出，并支持调用人脸识别（如问「我是谁」时用摄像头识别）。
推荐使用国内免费/低成本大模型：通义千问（阿里），新用户有免费额度。
"""
import os
import sys
import re

# 将项目根目录加入路径，以便导入 FaceRecognitionModule
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 语音识别（麦克风依赖 PyAudio，未装时仅支持 --text 模式）
try:
    import speech_recognition as sr
    import pyaudio  # noqa: F401
    HAS_PYAUDIO = True
except ImportError:
    sr = None
    HAS_PYAUDIO = False

# 通义千问
try:
    import dashscope
    from dashscope import Generation
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False

# 语音合成
try:
    import edge_tts
    import asyncio
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

# 人脸识别（调用 FaceRecognitionModule）
def _get_face_names():
    """调用人脸识别模块，返回当前摄像头画面中识别到的姓名列表。"""
    try:
        from FaceRecognitionModule.run_face_recognition import recognize_faces_from_camera
        names = recognize_faces_from_camera()
        return names if names else []
    except Exception as e:
        print(f"⚠️ 人脸识别调用失败: {e}")
        return []


# 判断用户是否在问「我是谁」类问题，需要调摄像头
def _need_face_context(text):
    if not text or not text.strip():
        return False
    t = text.strip()
    patterns = [
        r"我是谁", r"我是谁\s*[？?]?", r"看看我是谁", r"识别.*我",
        r"谁在(镜头|摄像头|画面)", r"你.*(认|识).*我", r"我叫什么",
        r"知道我是谁", r"认出我", r"镜头.*谁",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def _get_env_path():
    """.env 文件所在路径（与 speech_ai.py 同目录）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_api_key():
    """从环境变量或 .env 加载 API Key。"""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    env_path = _get_env_path()
    if not key and os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"\'')
                    if key and "xxxxxxxx" not in key:
                        break
                    key = ""
    return key


def listen_microphone(lang="zh-CN"):
    """从麦克风听一句话，返回识别到的文字，失败返回 None。"""
    if not HAS_PYAUDIO or sr is None:
        print("⚠️ 未安装 PyAudio，无法使用麦克风。请用  --text  模式输入文字，或先执行：brew install portaudio 再 pip install PyAudio")
        return None
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 请说话…")
        try:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source, duration=5)
        except Exception as e:
            print(f"⚠️ 录音失败: {e}")
            return None
    try:
        text = r.recognize_google(audio, language=lang)
        return text.strip() if text else None
    except OSError as e:
        if getattr(e, "errno", None) == 86:  # Bad CPU type in executable
            print("⚠️ Apple Silicon 需安装系统 flac，请执行: brew install flac")
            return None
        raise
    except sr.UnknownValueError:
        print("⚠️ 未识别到语音")
        return None
    except sr.RequestError as e:
        print(f"⚠️ 语音识别服务错误: {e}")
        return None


def chat_with_llm(user_text, face_context=None):
    """
    调用大模型得到回复。若提供 face_context，会作为当前「看到的人脸」注入系统提示。
    """
    api_key = _load_api_key()
    if not HAS_DASHSCOPE:
        return None, "未安装 dashscope，请执行: pip install dashscope"
    if not api_key:
        env_path = _get_env_path()
        return None, f"未配置 API Key。请在该文件填入通义千问 Key：{env_path}\n  （可复制 config_example.env 为 .env 后编辑）"

    dashscope.api_key = api_key
    system = "你是一个友好的语音助手，用简短口语化中文回答。"
    if face_context:
        system += f"\n【当前摄像头识别到的人脸】{face_context}。若用户问「我是谁」或类似问题，请根据上述信息回答。"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    try:
        resp = Generation.call(model="qwen-turbo", messages=messages)
        if resp.status_code == 200 and resp.output and resp.output.get("text"):
            return resp.output["text"].strip(), None
        return None, resp.message or "模型返回异常"
    except Exception as e:
        return None, str(e)


def speak_text(text, voice="zh-CN-YunxiNeural"):
    """用 edge-tts 朗读文本。"""
    if not text or not HAS_EDGE_TTS:
        return
    out = os.path.join(os.path.dirname(__file__), "_tts_out.mp3")

    async def _run():
        com = edge_tts.Communicate(text=text, voice=voice)
        await com.save(out)

    asyncio.run(_run())
    # 播放（macOS）
    if sys.platform == "darwin":
        os.system(f'afplay "{out}"')
    else:
        print(f"🔊 {text}")


def run_once(use_text_input=False):
    """单轮：听（或输入文字）-> 识别是否需要人脸 -> 调大模型 -> 说。"""
    if use_text_input:
        user = input("请输入文字（直接回车跳过）: ").strip()
    else:
        user = listen_microphone()
    if not user:
        return
    print(f"你说: {user}")
    face_context = None
    if _need_face_context(user):
        print("📷 正在用摄像头识别你的人脸…")
        names = _get_face_names()
        if names:
            face_context = "、".join(names)
            print(f"   识别到: {face_context}")
        else:
            face_context = "未识别到已知人脸"
    reply, err = chat_with_llm(user, face_context=face_context)
    if err:
        print(f"❌ {err}")
        speak_text("抱歉，我暂时无法回答。请检查网络和 API 配置。")
        return
    print(f"AI: {reply}")
    speak_text(reply)


def main():
    import argparse
    p = argparse.ArgumentParser(description="语音 AI 交互，支持问「我是谁」时调用人脸识别")
    p.add_argument("--text", action="store_true", help="使用文字输入代替麦克风（便于测试）")
    args = p.parse_args()

    print("---------- 语音 AI 交互 ----------")
    if not HAS_DASHSCOPE:
        print("请安装: pip install dashscope")
    if not HAS_EDGE_TTS:
        print("请安装: pip install edge-tts")
    env_path = _get_env_path()
    if not _load_api_key():
        print("请配置通义千问 API Key：")
        print(f"  1) 复制 config_example.env 为 .env（或创建 {env_path}）")
        print("  2) 在 https://dashscope.console.aliyun.com/ 创建 API Key")
        print("  3) 在 .env 中写一行：DASHSCOPE_API_KEY=sk-你的密钥")
    print("支持问「我是谁」：会调用摄像头人脸识别后回答。")
    if args.text:
        print("当前为文字输入模式（--text）。")
    elif not HAS_PYAUDIO:
        print("----------------------------------")
        print("❌ 未检测到 PyAudio，无法使用麦克风。")
        print("请先安装 PortAudio 再安装 PyAudio：")
        print("  macOS:  chmod +x install_mac.sh && ./install_mac.sh")
        print("  或:    brew install portaudio  然后  pip install PyAudio")
        print("----------------------------------")
        sys.exit(1)
    print("按 Ctrl+C 退出。")
    print("----------------------------------")
    use_text = args.text
    while True:
        try:
            run_once(use_text_input=use_text)
        except KeyboardInterrupt:
            print("\n再见。")
            break


if __name__ == "__main__":
    main()
