"""
语音 AI 交互：语音输入 -> 大模型 -> 语音输出，并支持调用人脸识别（如问「我是谁」时用摄像头识别）。
推荐使用国内免费/低成本大模型：通义千问（阿里），新用户有免费额度。
"""
import os
import sys
import re
import time
import cv2
import random
import subprocess
import signal
import threading
import queue
import shutil

# 将项目根目录加入路径，以便导入 FaceRecognitionModule
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 语音识别
try:
    import speech_recognition as sr
    import pyaudio  # noqa: F401
    HAS_PYAUDIO = True
except ImportError:
    sr = None
    HAS_PYAUDIO = False

# Whisper
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# 通义千问
try:
    import dashscope
    from dashscope import Generation, MultiModalConversation
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


# --- 辅助函数 ---

def _get_face_names():
    """调用人脸识别模块，返回当前摄像头画面中识别到的姓名列表。"""
    try:
        from FaceRecognitionModule.run_face_recognition import recognize_faces_from_camera
        names = recognize_faces_from_camera()
        return names if names else []
    except Exception as e:
        print(f"⚠️ 人脸识别调用失败: {e}")
        return []


def _need_face_context(text):
    if not text or not text.strip():
        return False
    t = text.strip()
    patterns = [
        r"我是谁", r"我是谁\s*[？?]?", r"看看我是谁", r"识别.*我",
        r"谁在(镜头|摄像头|画面)", r"你.*(认|识).*我", r"我叫什么",
        r"知道我是谁", r"认出我", r"镜头.*谁",
        r"(能|可以)?看(到|见)我吗", r"(能|可以)?看得到我吗", r"(能|可以)?看见我吗",
        r"看一?下?我", r"打开摄像头", r"识别一下我", r"看看有没有人",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def _need_vision_context(text):
    if not text:
        return False
    t = text.strip()
    if _need_face_context(t):
        return False
    patterns = [
        r"看.*(手里|拿|什么)", r"这是什么", r"描述.*(画面|场景|图片)",
        r"你看", r"帮我看", r"识别.*(物体|东西)", r"环境.*(怎么样|什么样)",
        r"读.*(文字|字)", r"摄像头.*(拍|看)",
        r"手里.*(拿|是).*", r".*拿的.*什么.*",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def _capture_image_file():
    """打开摄像头拍一张照片，保存为临时文件并返回路径。"""
    idx = -1
    for i in [1, 2, 0, 3]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            idx = i
            break
        cap.release()
    
    if idx < 0:
        print("⚠️ 未找到可用摄像头")
        return None

    print(f"📷 正在拍照 (Camera {idx})...")
    cap = cv2.VideoCapture(idx)
    # 预热
    for _ in range(15):
        cap.read()
        time.sleep(0.05)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("⚠️ 拍照失败")
        return None
    
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.close()
    cv2.imwrite(f.name, frame)
    return f.name


def _get_env_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_api_key():
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


def chat_with_llm(user_text, face_context=None, image_path=None, stream=False):
    """调用大模型得到回复。支持流式返回。"""
    api_key = _load_api_key()
    if not HAS_DASHSCOPE:
        return (None, "未安装 dashscope") if not stream else iter([])
    if not api_key:
        return (None, "未配置 API Key") if not stream else iter([])

    dashscope.api_key = api_key

    # 视觉多模态 (暂不支持流式，因为 MultiModalConversation 流式接口较复杂且 VL 生成较快)
    if image_path:
        if stream: print("⚠️ 视觉模式暂不支持流式，将转为一次性返回")
        print(f"🖼️ 正在调用视觉模型 (qwen-vl-max)...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{image_path}"},
                    {"text": user_text if user_text else "这张图里有什么？"}
                ]
            }
        ]
        try:
            resp = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
            if resp.status_code == 200:
                content_list = resp.output.choices[0].message.content
                text_reply = ""
                for item in content_list:
                    if "text" in item:
                        text_reply += item["text"]
                return (text_reply.strip(), None) if not stream else iter([text_reply.strip()])
            else:
                return (None, resp.message or "视觉模型异常") if not stream else iter([])
        except Exception as e:
            return (None, f"视觉模型失败: {e}") if not stream else iter([])

    # 文本/人脸模式
    system = "你是一个友好的语音助手，用简短口语化中文回答。"
    if face_context:
        system += f"\n【摄像头人脸】{face_context}。"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    
    try:
        if stream:
            # 流式调用
            # Qwen 的 stream=True 返回的是 iterator，每次 output.text 是全量文本(append模式)还是增量？
            # 经确认，qwen-turbo stream=True 时，output.text 是 *全量* 文本。需要自己 diff。
            # 但 generation 也有 incremental_output=True 选项 (部分模型支持)
            # 为兼容性，这里手动 diff
            def _generator():
                responses = Generation.call(model="qwen-turbo", messages=messages, result_format='message', stream=True, incremental_output=True)
                for resp in responses:
                    if resp.status_code == 200:
                        # incremental_output=True: output.choices[0].message.content 是增量
                        # 如果不支持 incremental，则 output.text 是全量。
                        # qwen-turbo 支持 incremental_output=True
                        content = resp.output.choices[0].message.content
                        if content:
                            yield content
                    else:
                        print(f"Model Error: {resp.message}")
            return _generator()
        else:
            resp = Generation.call(model="qwen-turbo", messages=messages)
            if resp.status_code == 200 and resp.output and resp.output.get("text"):
                return resp.output["text"].strip(), None
            return None, resp.message or "模型异常"
    except Exception as e:
        return (None, str(e)) if not stream else iter([])


# --- 核心类 ---

class AudioPlayer:
    """异步音频播放器，支持流式播放与打断。"""
    def __init__(self):
        self._play_thread = None
        self._stop_event = threading.Event()
        self._current_process = None
        self._temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
        # 清理旧缓存
        if os.path.exists(self._temp_dir):
            try: shutil.rmtree(self._temp_dir)
            except: pass
        os.makedirs(self._temp_dir, exist_ok=True)

    def play_stream(self, text_generator, voice="zh-CN-YunyangNeural", rate="+10%", pitch="-5Hz", blocking=True):
        """
        接收文本流（生成器），实时合成并播放。
        :param text_generator: 产出文本片段的生成器 (iterator)
        """
        if not HAS_EDGE_TTS: return

        # 停止上一次
        self.stop()
        self._stop_event.clear()

        # 队列
        q = queue.Queue()

        # 生产者：从 generator 读取文本 -> 缓冲 -> 按句切分 -> TTS -> 队列
        def producer():
            buffer = ""
            idx = 0
            
            # 正则：匹配标点符号
            split_pattern = r'([。！？；!?;]+)'
            
            try:
                for chunk in text_generator:
                    if self._stop_event.is_set(): break
                    if not chunk: continue
                    
                    buffer += chunk
                    
                    # 尝试切分
                    while True:
                        # 找第一个标点
                        match = re.search(split_pattern, buffer)
                        if not match:
                            break
                        
                        end_pos = match.end()
                        sentence = buffer[:end_pos]
                        buffer = buffer[end_pos:]
                        
                        clean = self._clean_text(sentence)
                        if clean:
                            _gen_audio(clean, idx)
                            idx += 1
                
                # 处理剩余文本
                if buffer and not self._stop_event.is_set():
                    clean = self._clean_text(buffer)
                    if clean:
                        _gen_audio(clean, idx)
            
            except Exception as e:
                print(f"⚠️ 流式处理异常: {e}")
            finally:
                q.put(None)

        def _gen_audio(text, i):
            filename = f"tts_stream_{int(time.time())}_{i}.mp3"
            filepath = os.path.join(self._temp_dir, filename)
            
            async def _run_tts():
                # 增加语速和音调参数
                com = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                await com.save(filepath)
            
            try:
                asyncio.run(_run_tts())
                q.put(filepath)
            except Exception as e:
                print(f"⚠️ TTS生成失败: {e}")

        # 消费者（同 play）
        def consumer():
            while not self._stop_event.is_set():
                try:
                    filepath = q.get(timeout=0.1)
                    if filepath is None: break
                except queue.Empty:
                    # 如果生产者活着，继续等；死了且空了，退出
                    if t_prod.is_alive(): continue
                    else: break
                
                if self._stop_event.is_set(): break

                # 播放
                if sys.platform == "darwin":
                    self._current_process = subprocess.Popen(
                        ['afplay', filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self._current_process.wait()
                    self._current_process = None
                else:
                    print(f"🔊 {filepath}...")

                try: os.remove(filepath)
                except: pass

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        
        t_prod.start()
        t_cons.start()

        if blocking:
            t_cons.join()
        else:
            self._play_thread = t_cons

    def play(self, text, voice="zh-CN-YunxiNeural", blocking=True):
        """流式播放语音。"""
        if not text or not HAS_EDGE_TTS:
            return
        self.stop()
        self._stop_event.clear()
        
        # 简单清洗
        clean_text = self._clean_text(text)
        if not clean_text:
            return

        # 预分段（按标点）
        parts = re.split(r'([。！？；!?;]+)', clean_text)
        chunks = []
        current_chunk = ""
        for p in parts:
            current_chunk += p
            if re.search(r'[。！？；!?;]', p):
                chunks.append(current_chunk)
                current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk)
            
        # 生产者：生成音频文件
        q = queue.Queue()
        
        def producer():
            for i, chunk in enumerate(chunks):
                if self._stop_event.is_set(): break
                if not chunk.strip(): continue
                
                filename = f"tts_{int(time.time())}_{i}.mp3"
                filepath = os.path.join(self._temp_dir, filename)
                
                async def _gen():
                    # 统一使用配置好的语速和音调
                    com = edge_tts.Communicate(text=chunk, voice=voice, rate="+10%", pitch="-5Hz")
                    await com.save(filepath)
                
                try:
                    asyncio.run(_gen())
                    q.put(filepath)
                except Exception as e:
                    print(f"⚠️ TTS 生成失败: {e}")
            q.put(None) # 结束标志

        # 消费者：播放
        def consumer():
            while not self._stop_event.is_set():
                try:
                    filepath = q.get(timeout=0.5)
                    if filepath is None: break
                except queue.Empty:
                    if not t_prod.is_alive() and q.empty():
                        break
                    continue
                
                if self._stop_event.is_set(): break
                
                if sys.platform == "darwin":
                    # Mac 使用 afplay
                    self._current_process = subprocess.Popen(
                        ['afplay', filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self._current_process.wait()
                    self._current_process = None
                else:
                    # Linux/Windows 暂略，假设 Mac
                    print(f"🔊 {text[:10]}...")
                
                # 播放完删除
                try:
                    os.remove(filepath)
                except:
                    pass

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        
        t_prod.start()
        t_cons.start()
        
        if blocking:
            t_cons.join()
        else:
            self._play_thread = t_cons

    def play_file(self, filepath, blocking=True):
        """播放本地文件（无延迟）。"""
        if not os.path.exists(filepath):
            return
        self.stop()
        self._stop_event.clear()
        
        if sys.platform == "darwin":
            self._current_process = subprocess.Popen(
                ['afplay', filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if blocking:
                self._current_process.wait()
                self._current_process = None
            else:
                # 如果非阻塞，我们需要一个线程来等待它结束（或者就不管了，但为了能stop，最好记录）
                # 简单起见，非阻塞模式下我们只记录 process，不join
                pass

    def stop(self):
        """停止播放。"""
        self._stop_event.set()
        
        # 终止当前播放进程
        if self._current_process:
            if self._current_process.poll() is None:
                self._current_process.terminate()
                try:
                    self._current_process.wait(timeout=0.1)
                except:
                    self._current_process.kill()
            self._current_process = None
        
        # 等待播放线程结束
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.2)

        try:
            for name in os.listdir(self._temp_dir):
                if name.endswith(".mp3"):
                    try:
                        os.remove(os.path.join(self._temp_dir, name))
                    except:
                        pass
        except:
            pass

    def is_playing(self):
        return self._play_thread and self._play_thread.is_alive()

    def _clean_text(self, text):
        if not text:
            return ""
        t = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        t = t.replace("*", "").replace("#", "").replace("`", "")
        return t.strip()


class SpeechAssistant:
    def __init__(self):
        self.r = sr.Recognizer()
        self.r.dynamic_energy_threshold = True
        self.r.pause_threshold = 0.8
        self.r.phrase_threshold = 0.4
        self.r.non_speaking_duration = 0.4
        
        self.mic = None
        self.whisper_model = None
        self.player = AudioPlayer()
        self._llm_stop_event = threading.Event()
        
        # 唤醒词
        self.WAKE_WORD = "小笨"
        
        # 状态
        self.is_active = False
        self.last_active_time = 0
        self.IDLE_TIMEOUT = 30
        
        self._init_mic()

    def _init_mic(self):
        """初始化麦克风与环境噪音。"""
        if self.mic: return
        print("🎤 初始化麦克风...")
        self.mic = sr.Microphone(sample_rate=16000, chunk_size=1024)
        with self.mic as source:
            # 仅校准一次
            print("🔇 正在校准环境噪音 (请保持安静 1秒)...")
            self.r.adjust_for_ambient_noise(source, duration=1)
            # 校准后关闭动态调整，防止 AI 说话时阈值漂移
            self.r.dynamic_energy_threshold = False
            # 固定阈值，避免因为扬声器导致阈值漂移
            self.r.energy_threshold = max(60, self.r.energy_threshold)
            print(f"✅ 校准完成 (阈值: {self.r.energy_threshold:.0f})")

    def _init_static_audio(self):
        """预生成静态提示音，消除网络延迟。"""
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_audio")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 定义需要缓存的静态文本
        static_texts = {
            "wake": "我在。",
            "listen": "请吩咐。",
            "bye": "好的，拜拜。",
            "rest": "那我先休息了，有事叫我。",
            "error": "抱歉，我没听清。"
        }
        
        self.static_audio_files = {}
        
        # 检查并生成
        for key, text in static_texts.items():
            filepath = os.path.join(cache_dir, f"{key}.mp3")
            self.static_audio_files[key] = filepath
            if not os.path.exists(filepath):
                print(f"🛠️ 生成静态音频: {text}")
                try:
                    # 使用与主语音一致的音色
                    async def _gen():
                        com = edge_tts.Communicate(text=text, voice="zh-CN-YunyangNeural", rate="+10%", pitch="-5Hz")
                        await com.save(filepath)
                    asyncio.run(_gen())
                except Exception as e:
                    print(f"⚠️ 生成静态音频失败: {e}")

    def _get_whisper(self):
        if not HAS_WHISPER: return None
        if not self.whisper_model:
            print("⏳ 加载 Whisper 模型 (tiny)...")
            try:
                import torch
                # 尝试使用 MPS (Metal Performance Shaders) 加速
                device = "cpu"
                if torch.backends.mps.is_available():
                    device = "mps"
                    print("🚀 使用 MPS 加速推理")
                self.whisper_model = whisper.load_model("tiny", device=device)
            except Exception as e:
                print(f"⚠️ MPS 加载失败，回退到 CPU: {e}")
                self.whisper_model = whisper.load_model("tiny", device="cpu")
        return self.whisper_model

    def listen(self, is_speaking=False):
        """监听并返回文本。支持打断检测。"""
        if not self.mic: return None
        
        # 优化打断：AI 说话时，使用极短的窗口(1s)进行切片监听
        phrase_limit = 0.8 if is_speaking else 8
        timeout = 0.6 if is_speaking else 6
        
        with self.mic as source:
            try:
                # pause_threshold: 说话后停顿多久算结束。
                # 正常对话 0.35s (更快)，打断时 0.18s (极速)
                self.r.pause_threshold = 0.18 if is_speaking else 0.35
                
                # non_speaking_duration: 多少秒静音算没人说话
                self.r.non_speaking_duration = 0.18
                
                audio = self.r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            except sr.WaitTimeoutError:
                return None
            except Exception as e:
                return None

        text = ""
        # 1. Whisper
        w_model = self._get_whisper()
        if w_model:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data())
                    tmp = f.name
                res = w_model.transcribe(
                    tmp,
                    language="zh",
                    fp16=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )
                text = res.get("text", "").strip()
                os.remove(tmp)
            except:
                pass
        
        # 2. Google fallback
        if not text:
            try:
                text = self.r.recognize_google(audio, language="zh-CN")
            except:
                pass
        
        return text.strip() if text else None

    def run(self):
        """主循环。"""
        self._init_mic()
        self._init_static_audio()
        
        print(f"\n✨ {self.WAKE_WORD} 语音助手已就绪 (Ctrl+C 退出)")
        print(f"👉 说 “{self.WAKE_WORD}” 唤醒我...")

        while True:
            try:
                # 检查是否超时休眠
                if self.is_active:
                    if time.time() - self.last_active_time > self.IDLE_TIMEOUT:
                        print("💤 超过30秒未交互，进入待机模式...")
                        self.player.play_file(self.static_audio_files.get("rest"), blocking=True)
                        self.is_active = False
                
                # 监听状态
                is_playing = self.player.is_playing()
                text = self.listen(is_speaking=is_playing)
                
                if not text:
                    continue
                
                # 打印听到的内容 (如果是播放中，可能听到自己，作为调试信息)
                if is_playing:
                    print(f"👂 [播放中监听] {text}")
                else:
                    print(f"👂 {text}")

                # 打断检测与自听过滤
                if is_playing:
                    # 模糊匹配唤醒词
                    is_wake = False
                    if self.WAKE_WORD in text:
                        is_wake = True
                    else:
                        # 同音词模糊匹配
                        fuzzy_words = ["小本", "校本", "晓笨", "小奔", "笨笨", "小蹦"]
                        for w in fuzzy_words:
                            if w in text:
                                is_wake = True
                                break
                    
                    if is_wake:
                        print(f"⚡️ 触发打断！")
                        self._llm_stop_event.set()
                        self.player.stop()
                        self.is_active = True
                        self.last_active_time = time.time()
                        # 立即播放本地缓存的“我在”，无延迟
                        self.player.play_file(self.static_audio_files.get("wake"), blocking=True)
                        continue
                    else:
                        # 只有听到唤醒词才算打断，否则视为自听（听到自己说话）
                        # print(f"🔇 忽略自听/背景音: {text}")
                        continue
                
                # 非播放状态的处理
                if not self.is_active:
                    # 待机模式：只响应唤醒词
                    if self.WAKE_WORD in text or "小本" in text or "笨笨" in text:
                        print("🚀 被唤醒！")
                        self.is_active = True
                        self.last_active_time = time.time()
                        # 立即播放本地缓存的“我在”
                        self.player.play_file(self.static_audio_files.get("wake"), blocking=True)
                else:
                    # 活跃模式
                    self.last_active_time = time.time()
                    
                    # 退出指令
                    if "再见" in text or "退下" in text or "休息" in text:
                        self.player.play_file(self.static_audio_files.get("bye"), blocking=True)
                        self.is_active = False
                        continue
                    
                    self._handle_command(text)
                    # 关键修改：AI 回复结束后，再次更新 last_active_time
                    # 这样休眠倒计时才会在 AI 说完后开始算
                    self.last_active_time = time.time()

            except KeyboardInterrupt:
                print("\n停止运行。")
                break
            except Exception as e:
                print(f"❌ 主循环错误: {e}")
                time.sleep(1)

    def _handle_command(self, text):
        self._llm_stop_event.clear()
        # 视觉
        img_path = None
        if _need_vision_context(text):
            self.player.play("好的，我看看。", blocking=False)
            img_path = _capture_image_file()
        
        # 人脸
        face_ctx = None
        if not img_path and _need_face_context(text):
            self.player.play("正在识别...", blocking=False)
            names = _get_face_names()
            if names:
                known = [n for n in names if n != "未知"]
                if known: face_ctx = "、".join(known)
        
        # LLM 流式调用
        # 注意：如果有 img_path，目前 chat_with_llm 会自动降级为非流式返回 list
        stream_gen = chat_with_llm(text, face_context=face_ctx, image_path=img_path, stream=True)
        if img_path and os.path.exists(img_path): os.remove(img_path)
        
        print("AI: ", end="", flush=True)
        
        # 包装生成器以打印输出
        def _printing_gen():
            for chunk in stream_gen:
                if self._llm_stop_event.is_set():
                    break
                print(chunk, end="", flush=True)
                yield chunk
            print("") # 换行

        # 播放流式音频
        # 使用 zh-CN-YunyangNeural (新闻男声) + rate="+10%" (自然语速) + pitch="-5Hz" 模拟沉稳贾维斯风格
        self.player.play_stream(_printing_gen(), voice="zh-CN-YunyangNeural", rate="+10%", pitch="-5Hz", blocking=False)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true")
    args = p.parse_args()
    
    if args.text:
        print("文字模式...")
        while True:
            t = input("输入: ")
            reply, _ = chat_with_llm(t)
            print(f"AI: {reply}")
    else:
        app = SpeechAssistant()
        app.run()

if __name__ == "__main__":
    main()
