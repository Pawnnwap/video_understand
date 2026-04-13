"""
Generate a synthetic ~60s test video with Chinese TTS speech + colored slides.
Requires: edge-tts, ffmpeg
"""
import asyncio
import subprocess
import sys
from pathlib import Path

SPEECH_TEXTS = [
    "你好，欢迎来到视频理解系统测试。",
    "今天我们来介绍一下人工智能的基本概念。",
    "人工智能是指由人类制造的机器所表现出来的智能。",
    "机器学习是人工智能的一个重要分支。",
    "深度学习使用多层神经网络来学习数据的特征表示。",
    "自然语言处理让计算机能够理解和生成人类语言。",
    "计算机视觉让机器能够理解和分析图像和视频。",
    "感谢您使用本系统进行测试，希望一切顺利！",
]

async def gen_speech(text: str, out_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(out_path)

async def main():
    tmp = Path("test_tmp")
    tmp.mkdir(exist_ok=True)

    print("Generating TTS speech segments...")
    mp3_files = []
    for i, text in enumerate(SPEECH_TEXTS):
        mp3_path = str(tmp / f"seg_{i:02d}.mp3")
        await gen_speech(text, mp3_path)
        mp3_files.append(mp3_path)
        print(f"  [{i+1}/{len(SPEECH_TEXTS)}] {text[:20]}...")

    # Concatenate all mp3 files into one wav
    print("Merging audio...")
    list_file = tmp / "mp3_list.txt"
    # Use absolute paths with forward slashes for ffmpeg concat list
    with open(list_file, "w", encoding="utf-8") as f:
        for mp3 in mp3_files:
            abs_path = str(Path(mp3).resolve()).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    audio_out = tmp / "speech.wav"
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-ar", "16000", "-ac", "1",
        str(audio_out)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg concat stderr:", result.stderr[-500:])
        raise RuntimeError(f"ffmpeg concat failed: {result.returncode}")

    # Get audio duration
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_out)
    ], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f"Audio duration: {duration:.1f}s")

    # Create a simple test video: colored background with text overlay
    print("Creating test video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:size=1280x720:duration={duration}:rate=25",
        "-i", str(audio_out),
        "-vf", (
            "drawtext=text='AI Video Understanding Test':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=100,"
            "drawtext=text='FunASR + LM Studio + PaddleOCR':fontcolor=cyan:fontsize=32:x=(w-text_w)/2:y=200,"
            "drawtext=text='Smoke Test Video':fontcolor=yellow:fontsize=36:x=(w-text_w)/2:y=320"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-shortest",
        "test_video.mp4"
    ], check=True, capture_output=True)

    duration_final = float(subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", "test_video.mp4"
    ], capture_output=True, text=True).stdout.strip())

    print(f"Test video created: test_video.mp4 ({duration_final:.1f}s)")

    # Cleanup tmp
    import shutil
    shutil.rmtree(tmp)

if __name__ == "__main__":
    asyncio.run(main())
