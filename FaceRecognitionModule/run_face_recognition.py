"""
人脸识别入口：从摄像头实时检测并识别已知人脸。
将已知人物照片放入 known_faces 文件夹，文件名即显示名称（如 张三.jpg）。

注意：请运行本文件（run_face_recognition.py），不要将脚本命名为 face_recognition.py，
否则会与已安装的 face_recognition 库冲突导致无法识别。
"""
import cv2
import sys
import os
import numpy as np

# 使用已安装的 face_recognition 库（脚本名必须不是 face_recognition.py）
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "known_faces")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# 性能与识别参数
DETECT_SCALE = 0.25  # 检测时缩小到 25% 分辨率，大幅提升帧率
DETECT_EVERY_N = 2   # 每 N 帧做一次人脸检测，中间帧复用结果
MATCH_THRESHOLD = 0.65  # 识别阈值，越大越宽松（易认出但易误识）


def _get_chinese_font(size=24):
    if not HAS_PIL:
        return None
    for name in ("PingFang.ttc", "SimHei.ttf", "msyh.ttc", "Arial Unicode.ttf"):
        for base in ("/System/Library/Fonts", "/Library/Fonts", "C:/Windows/Fonts", "/usr/share/fonts"):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size, encoding="utf-8")
                except Exception:
                    pass
    return ImageFont.load_default()


def _draw_text(frame, text, left, top, color_bgr=(0, 255, 0)):
    if not text:
        return
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    if any(ord(c) > 127 for c in text) and HAS_PIL:
        font = _get_chinese_font(28)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        y = max(10, top - 5)
        draw.text((left, y), text, color_rgb, font=font)
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text, (left, max(25, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)


def get_camera_index():
    order = [1, 2, 0, 3]
    for index in order:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return -1


def load_known_faces():
    known_encodings = []
    known_names = []
    if not HAS_FACE_RECOGNITION:
        return known_encodings, known_names
    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return known_encodings, known_names
    exts = (".jpg", ".jpeg", ".png")
    for name in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isfile(path) or not name.lower().endswith(exts):
            continue
        label = os.path.splitext(name)[0]
        try:
            img = face_recognition.load_image_file(path)
            # 用 num_jitters=2 得到更稳的编码，便于识别成功
            encodings = face_recognition.face_encodings(img, num_jitters=2)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(label)
        except Exception as e:
            print(f"⚠️ 加载已知人脸失败 {name}: {e}")
    return known_encodings, known_names


def _get_haar_cascade_path():
    """获取 Haar 级联路径：优先 OpenCV 自带，否则使用项目 data 目录下的文件"""
    candidates = []
    if getattr(cv2, "data", None) is not None:
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    cv2_dir = os.path.dirname(cv2.__file__)
    candidates.extend([
        os.path.join(cv2_dir, "data", "haarcascades", "haarcascade_frontalface_default.xml"),
        os.path.join(cv2_dir, "..", "cv2", "data", "haarcascades", "haarcascade_frontalface_default.xml"),
    ])
    # 项目内回退路径（无完整 OpenCV 时使用）
    os.makedirs(DATA_DIR, exist_ok=True)
    candidates.append(os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml"))
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
    return candidates[-1] if candidates else ""


def detect_faces_opencv(frame):
    cascade_path = _get_haar_cascade_path()
    if not cascade_path or not os.path.isfile(cascade_path):
        return []
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    return faces


def _draw_cached_boxes(frame, boxes_and_names):
    """仅在画面上绘制已缓存的人脸框，不做检测。boxes_and_names: [((left,top,right,bottom), name), ...]"""
    for (left, top, right, bottom), name in boxes_and_names:
        color = (0, 0, 255) if name == "未知" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        _draw_text(frame, name, left, top, color)


def process_frame(frame, known_encodings, known_names):
    """返回 (处理后的 frame, 本帧的人脸框与姓名列表，供跳帧时复用)。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_full, w_full = rgb.shape[:2]
    boxes_and_names = []

    def draw_face_box(left, top, right, bottom, name, color_bgr):
        cv2.rectangle(frame, (left, top), (right, bottom), color_bgr, 2)
        _draw_text(frame, name, left, top, color_bgr)

    if HAS_FACE_RECOGNITION:
        w_small = max(160, int(w_full * DETECT_SCALE))
        h_small = max(120, int(h_full * DETECT_SCALE))
        small = cv2.resize(rgb, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = w_full / w_small, h_full / h_small

        face_locations = face_recognition.face_locations(small, model="hog", number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(small, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            left_f = int(left * scale_x)
            right_f = int(right * scale_x)
            top_f = int(top * scale_y)
            bottom_f = int(bottom * scale_y)
            name = "未知"
            encoding = face_encodings[i] if i < len(face_encodings) else None
            if encoding is not None and known_encodings and known_names:
                distances = face_recognition.face_distance(known_encodings, encoding)
                idx = int(np.argmin(distances))
                if distances[idx] < MATCH_THRESHOLD:
                    name = known_names[idx]
            color = (0, 0, 255) if name == "未知" else (0, 255, 0)
            draw_face_box(left_f, top_f, right_f, bottom_f, name, color)
            boxes_and_names.append(((left_f, top_f, right_f, bottom_f), name))
        if not face_locations:
            faces = detect_faces_opencv(frame)
            for (x, y, w, h) in faces:
                draw_face_box(x, y, x + w, y + h, "未知", (0, 0, 255))
                boxes_and_names.append(((x, y, x + w, y + h), "未知"))
    else:
        faces = detect_faces_opencv(frame)
        for (x, y, w, h) in faces:
            draw_face_box(x, y, x + w, y + h, "Face", (0, 255, 0))
    return frame, boxes_and_names


def recognize_faces_from_frame(frame, known_encodings, known_names):
    """
    对单帧画面做人脸识别，返回识别到的姓名列表（供其他模块如语音 AI 调用）。
    若未识别到任何人或未匹配到已知人脸，列表中为 "未知" 或为空。
    """
    if not HAS_FACE_RECOGNITION:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_full, w_full = rgb.shape[:2]
    w_small = max(160, int(w_full * DETECT_SCALE))
    h_small = max(120, int(h_full * DETECT_SCALE))
    small = cv2.resize(rgb, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    scale_x, scale_y = w_full / w_small, h_full / h_small
    face_locations = face_recognition.face_locations(small, model="hog", number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(small, face_locations)
    names = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        name = "未知"
        encoding = face_encodings[i] if i < len(face_encodings) else None
        if encoding is not None and known_encodings and known_names:
            distances = face_recognition.face_distance(known_encodings, encoding)
            idx = int(np.argmin(distances))
            if distances[idx] < MATCH_THRESHOLD:
                name = known_names[idx]
        names.append(name)
    return names


def recognize_faces_from_camera(known_faces_dir=None):
    """
    打开摄像头拍一帧并做人脸识别，返回当前画面中识别到的姓名列表。
    供 AISpeechInteraction 等模块调用，例如用户问「我是谁」时调用此函数。
    known_faces_dir: 已知人脸目录，默认使用 FaceRecognitionModule/known_faces。
    """
    if known_faces_dir is None:
        known_faces_dir = KNOWN_FACES_DIR
    # 临时改用指定目录加载已知人脸（若与当前 SCRIPT_DIR 一致则用现有逻辑）
    if os.path.normpath(known_faces_dir) != os.path.normpath(KNOWN_FACES_DIR):
        known_encodings, known_names = [], []
        if HAS_FACE_RECOGNITION and os.path.isdir(known_faces_dir):
            exts = (".jpg", ".jpeg", ".png")
            for name in os.listdir(known_faces_dir):
                path = os.path.join(known_faces_dir, name)
                if not os.path.isfile(path) or not name.lower().endswith(exts):
                    continue
                label = os.path.splitext(name)[0]
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img, num_jitters=2)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(label)
                except Exception:
                    pass
    else:
        known_encodings, known_names = load_known_faces()
    idx = get_camera_index()
    if idx < 0:
        return []
    cap = cv2.VideoCapture(idx)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return []
    return recognize_faces_from_frame(frame, known_encodings, known_names)


def main():
    print("---------- 依赖检查 ----------")
    cascade_path = _get_haar_cascade_path()
    cascade_ok = bool(cascade_path and os.path.isfile(cascade_path))
    print(f"  face_recognition 库: {'已安装' if HAS_FACE_RECOGNITION else '未安装'}")
    print(f"  OpenCV 人脸模型: {'已找到' if cascade_ok else '未找到'}")
    if cascade_path:
        print(f"    路径: {cascade_path}")
    if not cascade_ok:
        print("  ⚠️ 请执行以下任一方式：")
        print("     1) pip uninstall opencv-python-headless; pip install opencv-python")
        print("     2) 或下载模型到 data 目录: 见 README 或运行 download_cascade.py")

    if not HAS_FACE_RECOGNITION:
        print("💡 安装人脸识别库: pip install face_recognition")
    known_encodings, known_names = load_known_faces()
    if known_encodings:
        print(f"✅ 已加载 {len(known_names)} 个已知人脸: {', '.join(known_names)}")
    else:
        print("💡 在 known_faces 中放入已知人物照片即可识别")
    print("------------------------------")

    camera_idx = get_camera_index()
    if camera_idx == -1:
        print("❌ 未检测到可用摄像头")
        sys.exit(1)

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win_name = "Face Recognition (q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    for _ in range(5):
        cap.read()

    print(f"✅ 成功连接摄像头，索引：{camera_idx}")
    print("💡 按 'q' 键退出程序")

    first_frame_diagnostic_done = False
    frame_count = 0
    cached_boxes = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ 无法读取摄像头画面")
            break
        try:
            if not first_frame_diagnostic_done and frame is not None:
                n_cv = len(detect_faces_opencv(frame))
                n_hog = 0
                if HAS_FACE_RECOGNITION:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    n_hog = len(face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1))
                print(f"📷 首帧检测: face_recognition={n_hog} 人脸, OpenCV={n_cv} 人脸")
                first_frame_diagnostic_done = True
            # 每 DETECT_EVERY_N 帧做一次完整检测，中间帧只复用上一帧的框，提升帧率
            if frame_count % DETECT_EVERY_N == 0 or not cached_boxes:
                frame, cached_boxes = process_frame(frame, known_encodings, known_names)
            else:
                _draw_cached_boxes(frame, cached_boxes)
            frame_count += 1
        except Exception as e:
            print(f"⚠️ 处理画面时出错: {e}")
        h, w = frame.shape[:2]
        if w < 640 or h < 480:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
