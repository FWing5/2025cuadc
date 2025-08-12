import time
import argparse
from yololite_detector import YOLOv5LiteDetector
from vision_utils import Camera, send_frame

def main(if_send: bool, send_freq: float):
    detector = YOLOv5LiteDetector(view_img=False)
    cam = Camera()

    print("按 Ctrl+C 退出。")

    # 预热推理，减少第一次时间误差
    for _ in range(5):
        img = cam.read()
        _ = detector.detect(img)

    last_send_time = 0
    try:
        while True:
            frame = cam.read()
            start = time.time()
            detections, result_img = detector.detect(frame)
            end = time.time()

            infer_time = (end - start) * 1000  # ms
            print(f"推理时间: {infer_time:.2f} ms, 检测到目标数: {len(detections)}")

            if if_send:
                now = time.time()
                # send_freq == 0 表示每帧都发送
                if send_freq == 0 or (now - last_send_time) >= send_freq:
                    send_frame(frame, cam.url)
                    last_send_time = now
    except KeyboardInterrupt:
        print("退出检测。")
    finally:
        cam.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5Lite Camera 推理速度及发送测试")
    parser.add_argument('--if_send', type=int, default=0, help="是否发送帧，1表示发送，0不发送")
    parser.add_argument('--send_freq', type=float, default=0, help="发送频率（秒），0表示每帧都发送")

    args = parser.parse_args()
    main(if_send=bool(args.if_send), send_freq=args.send_freq)
