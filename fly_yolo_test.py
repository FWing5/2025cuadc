import time
import select
import sys
from drone_controller import DroneController
from vision_utils import *
from yololite_detector import YOLOv5LiteDetector

CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_OFFSET_THRESHOLD = 50  # 像素阈值
LOOP_HZ = 5  # 控制循环频率（次/秒）

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr

drone = DroneController()
detector = YOLOv5LiteDetector(view_img=False)
cam = Camera(CAM_WIDTH, CAM_HEIGHT)

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

last_time = time.time()

while True:
    if key_pressed():
        key = sys.stdin.readline().strip()
        if key == 'q': 
            print("退出程序")
            break

    frame = cam.read()

    # 控制循环频率
    now = time.time()
    if now - last_time < 1.0 / LOOP_HZ:
        continue
    last_time = now

    # 目标检测
    dtc, img = detector.detect(frame)
    send_frame(img)

    if not dtc:
        print("[WARN] 未检测到目标")
        continue

    print(f"[DEBUG] 检测到 {len(dtc)} 个目标")

    _, _, cur_z = drone.get_position()
    print(f"[DEBUG] 当前高度: {cur_z:.2f} m")

    if cur_z <= 1.1:
        print("[INFO] 进入圆形检测阶段...")
        circles = detect_circles(frame)
        if circles is not None:
            offset = get_circle_offset_in_closest_bbox(dtc, circles, CAM_WIDTH, CAM_HEIGHT)
            if offset:
                dx, dy = offset
                print(f"[DEBUG] 圆心偏移 dx={dx:.2f}, dy={dy:.2f}")
                if abs(dx) < TARGET_OFFSET_THRESHOLD and abs(dy) < TARGET_OFFSET_THRESHOLD:
                    print("[ACTION] 投放物品")
                    drone.drop_front()
                    break
                else:
                    print("[ACTION] 调整位置以对准圆心")
                    drone.adjust_position_by_pixel_offset(dx, dy)
        else:
            print("[WARN] 未检测到圆形")

    else:
        dx, dy = get_closest_center_offset(dtc, CAM_WIDTH, CAM_HEIGHT)
        print(f"[DEBUG] 目标中心偏移 dx={dx:.2f}, dy={dy:.2f}")
        if abs(dx) < TARGET_OFFSET_THRESHOLD and abs(dy) < TARGET_OFFSET_THRESHOLD:
            print("[ACTION] 下降 0.5m")
            drone.down(0.5, 1)
        else:
            print("[ACTION] 调整位置以对准目标框中心")
            drone.adjust_position_by_pixel_offset(dx, dy)

print("[INFO] 返回起始点并降落")
drone.fly_to(0, 0, -4, 10)
drone.land()

time.sleep(10)
drone.shutdown()
print("[INFO] 程序结束")
