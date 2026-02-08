"""将 OpenCV 人脸检测模型下载到 data 目录，解决「OpenCV 人脸模型: 未找到」"""
import os
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
OUT = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"正在下载: {URL}")
    try:
        urllib.request.urlretrieve(URL, OUT)
        print(f"✅ 已保存到: {OUT}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("  请手动下载上述 URL 并保存到 data 目录下同名文件。")

if __name__ == "__main__":
    main()
