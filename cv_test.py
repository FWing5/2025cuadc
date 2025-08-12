import time
import select
import sys
import cv2
from vision_utils import *
from yolo_detector import YOLOv5Detector
from drone_controller import DroneController

CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_OFFSET_THRESHOLD = 50  # 像素阈值
LOOP_HZ = 5  # 控制循环频率（次/秒）

def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr


def main_loop():
    flag = 2
    last_time = time.time()

    cam = Camera()
    detector = YOLOv5Detector(view_img=False)

    quad = None
    perspective_matrix = None

    while True:
        if key_pressed():
            key = sys.stdin.readline().strip()
            if key == 'q': 
                print("退出程序")
                break

        # 控制循环频率
        now = time.time()
        if now - last_time < 1.0 / LOOP_HZ:
            continue
        last_time = now

        frame = cam.read()

        if flag == 2 :
            if frame is None:
                print("[WARN] 读帧失败，跳过本次循环")
                continue

            # 查找最大四边形
            quad_found = find_largest_quadrilateral(frame)
            if quad_found is None:
                print("未识别到四边形，重试")
                time.sleep(1)
                continue
            quad = quad_found

            # YOLO识别目标且筛选四边形内的目标
            targets_inside, debug_img = yolo_detection_on_quad(frame, quad, detector)

            if len(targets_inside) == 3:
                # 按cx排序，方便后续逻辑
                targets_inside.sort(key=lambda x: x[0][0])

                print("识别成功，目标中点坐标（左到右）：")
                for i, (pt, cls) in enumerate(targets_inside):
                    print(f"目标{i+1}: 中心 = {pt}, 类别 = {cls}")

                quad = sort_quad_points(quad)

                # 这里你给的实际世界四边形坐标，单位米？
                real_quad = [
                    (38, -4),  # 图像左上 → 世界右上
                    (38, 4),   # 图像右上 → 世界右下
                    (32, 4),   # 图像右下 → 世界左下
                    (32, -4),  # 图像左下 → 世界左上
                ]
                perspective_matrix = compute_perspective_transform(quad, real_quad)

                target_world_coords = []
                for (cx, cy), cls in targets_inside:
                    wx, wy = pixel_to_world(cx, cy, perspective_matrix)
                    target_world_coords.append((wx, wy))
                print("目标世界坐标：", target_world_coords)

                flag =3  # 进入下一状态
            else:
                print(f"当前目标数量为 {len(targets_inside)}，未满足3个，重试中...")

            # 画出四边形边界
            for i in range(4):
                cv2.line(debug_img, tuple(quad[i]), tuple(quad[(i + 1) % 4]), (255, 0, 0), 2)
            # 显示调试图像（可选）
            send_frame(debug_img)

        else:
            break


drone = DroneController()
detector = YOLOv5Detector(view_img=False)
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

main_loop()

print("[INFO] 返回起始点并降落")
drone.fly_to(0, 0, -4, 10)
drone.land()

time.sleep(10)
drone.shutdown()
print("[INFO] 程序结束")
