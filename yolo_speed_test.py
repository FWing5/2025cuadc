import time
import select
import sys
from drone_controller import DroneController
from vision_utils import *
from yolo_detector import YOLOv5Detector

CAM_WIDTH = 640
CAM_HEIGHT = 480
LOOP_HZ = 5  # 控制循环频率（次/秒）

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr

drone = DroneController()
detector = YOLOv5Detector(view_img=False)
cam = Camera(CAM_WIDTH, CAM_HEIGHT)

params_list = [
    {"img_size": 480},
    {"img_size": 416},
    {"img_size": 320}
]

# 启动前拍一帧
frame = cam.read()
send_frame(frame)

print("[INFO] 起飞到巡航高度...")
drone.fly_to(0, 0, -4, 10)
send_frame(cam.read())

# 预热 YOLO
print("[INFO] 正在预热检测器...")
for _ in range(5):
    frame = cam.read()
    _ = detector.detect(frame)
print("[INFO] 检测器预热完成")

for idx, params in enumerate(params_list):
    print(f"\n=== 开始第 {idx+1} 轮测试，参数: {params} ===")
    detector = YOLOv5Detector(view_img=False, **params)

    start_time = time.time()
    while time.time() - start_time < 10:  # 每轮测试 10 秒
        frame = cam.read()

        t0 = time.time()
        detections, result_img = detector.detect(frame)
        t1 = time.time()

        infer_time = (t1 - t0) * 1000
        print(f"推理时间: {infer_time:.2f} ms, 检测到目标数: {len(detections)}")

        if len(detections) == 3:
            try:
                send_frame(result_img, cam.url)
            except Exception as e:
                print(f"跳过执行，原因: {e}")

            


cam.release()

print("[INFO] 降落")
drone.land()

time.sleep(10)
drone.shutdown()
print("[INFO] 程序结束")
