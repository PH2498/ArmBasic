#!/bin/bash
# macOS 一键安装依赖（含麦克风 PyAudio）
# 必须先装 PortAudio，否则 PyAudio 编译会报错 portaudio.h not found

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== 1. 安装系统依赖（Homebrew）=========="
if command -v brew &>/dev/null; then
    brew install portaudio   # PyAudio 依赖
    brew install flac        # 语音识别转码，Apple Silicon 必须用系统 flac
else
    echo "未找到 Homebrew。请先安装: https://brew.sh"
    echo "安装后重新运行: ./install_mac.sh"
    exit 1
fi

echo ""
echo "========== 2. 安装 Python 依赖 =========="
pip install -r requirements.txt

echo ""
echo "✅ 安装完成。运行: python speech_ai.py"
