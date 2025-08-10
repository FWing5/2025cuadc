from vision_utils import Camera, send_frame
import time
import signal
import sys

camera = Camera()

def cleanup_and_exit(signum, frame):
    print("收到退出信号，释放摄像头资源...")
    camera.release()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)   # 捕获 Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # 捕获 kill 信号

try:
    print("正在每1s发送一帧画面")
    while True:
        time.sleep(1)
        frame = camera.read()
        send_frame(frame)
except Exception as e:
    print(f"异常退出: {e}")
    camera.release()
