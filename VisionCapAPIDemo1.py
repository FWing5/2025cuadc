import time
import math
import numpy as np
import cv2
import sys
import torch
from collections import deque

# 导入YOLOv5-lite模块
sys.path.insert(0, './YOLOv5-Lite')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# 控制接口与图像采集接口
import PX4MavCtrlV4 as PX4MavCtrl
import VisionCaptureApi
import UE4CtrlAPI

# 初始化UE4与视觉模块
ue = UE4CtrlAPI.UE4CtrlAPI()
vis = VisionCaptureApi.VisionCaptureApi()
vis.jsonLoad()
if not vis.sendReqToUE4():
    sys.exit(0)
vis.startImgCap(True)

# 设置分辨率与帧率
ue.sendUE4Cmd('r.setres 1080x720w', 0)
ue.sendUE4Cmd('t.MaxFPS 30', 0)
time.sleep(2)

# 初始化PX4控制器
mav = PX4MavCtrl.PX4MavCtrler(1)
mav.InitMavLoop()

# 初始化时间控制
startTime = time.time()
timeInterval = 1 / 30.0
lastTime = time.time()

# 加载YOLOv5模型
device = select_device('')
weights_path = './YOLOv5-Lite/weights/best.pt'
model = attempt_load(weights_path, map_location=device)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names
half = device.type != 'cpu'
if half:
    model.half()


# 控制流程与状态标记
flag = 0
num = 0
lastClock = time.time()
target_history = deque(maxlen=30)
hover_start_time = None
target_stage = 0
init34 = 0

# 位置记录
home_position = (0, 0, -4.5)
search_position = (34.5, 0, -4.5)
hover_position = (59.5, 0, -4.5)

# 获取图像帧
def get_current_frame():
    if vis.hasData[0]:
        frame = vis.Img[0]
        if frame is not None and frame.size > 0:
            return frame
    return None

# 获取目标网格key（用于识别历史一致目标）
def get_target_key(cls, cx, cy):
    return f"{cls}_{cx//10}_{cy//10}"

# 四边形检测
def find_largest_quadrilateral(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_quad = approx

    if best_quad is not None:
        return best_quad.reshape(4, 2)
    return None

def compute_perspective_transform(image_quad, world_quad):
    # 输入：图像中的四边形（像素），和真实坐标系下的四边形（米级）
    pts_img = np.float32(image_quad)
    pts_real = np.float32(world_quad)
    matrix = cv2.getPerspectiveTransform(pts_img, pts_real)
    return matrix

def pixel_to_world(px, py, transform_matrix):
    pts = np.array([[ [px, py] ]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, transform_matrix)
    return tuple(transformed[0][0])

def sort_quad_points(pts):
    # pts: 4x2 array
    # 返回顺时针排序：TL, TR, BR, BL
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sorted_idx = np.argsort(angles)
    return pts[sorted_idx]

# 计算点是否在四边形内（用cv2的点多边形测试）
def point_in_quad(point, quad):
    return cv2.pointPolygonTest(quad.astype
    (np.int32), point, False) >= 0

def yolo_window():
    frame = get_current_frame()
    if frame is not None:
        img = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img0 = frame.copy()  # 原图
        img = letterbox(img0, new_shape=640, stride=32, auto=True)[0]
        img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if model.half:  # 只有模型使用 half 才转换
            img = img.half()

        pred = model(img)[0]  # 官方版 DetectMultiBackend() 不需要 [0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        frame_targets = []
        if pred and len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                key = get_target_key(int(cls), cx, cy)
                area = (x2 - x1) * (y2 - y1)
                frame_targets.append((key, area, cx, cy, int(cls)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        target_history.append(frame_targets)
        cv2.imshow("YOLOv5 Detection", frame)
        cv2.waitKey(1)
        return frame

def drone_hover(duration_sec):
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        yolo_window()
        cur_x, cur_y, cur_z = mav.uavPosNED
        mav.SendPosNED(cur_x, cur_y, cur_z, 0)
        time.sleep(0.05)  # 控制频率，200ms发送一次

def drone_fly_with_camera(x, y, z, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        yolo_window()
        mav.SendPosNED(x, y, z, 0)
        time.sleep(0.05)  # 控制频率，200ms发送一次

# 主循环逻辑
while True:
    lastTime += timeInterval
    sleepTime = lastTime - time.time()
    if sleepTime > 0:
        time.sleep(sleepTime)
    else:
        lastTime = time.time()

    frame = yolo_window()

    num += 1
    if num % 100 == 0:
        now = time.time()
        print("MainThreadFPS:", 100 / (now - lastClock))
        lastClock = now

    now = time.time()

    # 状态控制流程
    if flag == 0 and now - startTime > 5:
        print("5s: Arm & takeoff to 5m")
        mav.initOffboard()
        mav.SendMavArm(True)
        mav.SendPosNED(*home_position, 0)
        flag = 1

    if flag == 1 and now - startTime > 15:
        print("15s: Move to pre-search area")
        mav.SendCopterSpeed(3)
        drone_fly_with_camera(*search_position, 15)
        flag = 2

    if flag == 2 and now - startTime > 32:
        yolo_window()
        if frame is None:
            continue

        # 尝试找最大四边形（只识别一次，成功就进入下一个状态）
        quad = find_largest_quadrilateral(frame)
        if quad is None:
            print("未识别到四边形，重试")
            continue

        # 用YOLO识别目标
        img = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img0 = frame.copy()  # 原图
        img = letterbox(img0, new_shape=640, stride=32, auto=True)[0]
        img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if model.half:  # 只有模型使用 half 才转换
            img = img.half()

        pred = model(img)[0]  # 官方版 DetectMultiBackend() 不需要 [0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        targets_inside = []

        if pred and len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if point_in_quad((cx, cy), quad):
                    targets_inside.append(((cx, cy), int(cls)))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if len(targets_inside) == 3:
            # 按cx从小到大排序
            targets_inside.sort(key=lambda x: x[0][0])
            print("识别成功，目标中点坐标（左到右）：")
            for i, (pt, cls) in enumerate(targets_inside):
                print(f"目标{i+1}: 中心 = {pt}, 类别 = {cls}")

            quad = sort_quad_points(quad)
            real_quad = [
                (38, -4),  # 图像左上 → 世界右上
                (38, 4),   # 图像右上 → 世界右下
                (32, 4),   # 图像右下 → 世界左下
                (32, -4),  # 图像左下 → 世界左上
            ]
            perspective_matrix = compute_perspective_transform(quad, real_quad)

            # 对YOLO识别到的3个目标中心像素位置进行坐标映射
            target_world_coords = []
            for (cx, cy), cls in targets_inside:
                wx, wy = pixel_to_world(cx, cy, perspective_matrix)
                target_world_coords.append((wx, wy))

            flag = 3
            continue
        else:
            print(f"当前目标数量为 {len(targets_inside)}，未满足3个，重试中...")

        # 显示调试图像
        for i in range(4):
            cv2.line(frame, tuple(quad[i]), tuple(quad[(i + 1) % 4]), (255, 0, 0), 2)

    if flag in [3, 4]:
        target_pos = target_world_coords[flag - 3]
        base_height = -2.0 + 0.25 * target_stage

        if init34 == 0:
            x, y = target_pos
            drone_fly_with_camera(x, y, base_height, 3)
            init34 = 1

        cur_x, cur_y, cur_z = mav.uavPosNED

        if target_stage >= 0:
            drone_fly_with_camera(cur_x, cur_y, base_height, 1.5)
            print(f"飞往 {cur_x}, {cur_y}, {base_height}")

            found = False
            for _ in range(60):
                frame = yolo_window()
                if frame is None:
                    continue

                img = cv2.resize(frame, (640, 640))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).to(device).permute(2, 0, 1).float() / 255.0
                if img_tensor.ndimension() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                pred = model(img_tensor)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

                if pred and len(pred[0]):
                    det = pred[0]
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                    # ✅ 1. 多目标情况：直接下降
                    if len(det) >= 2:
                        print("检测到多个目标，直接下降 0.25m")
                        target_stage += 1
                        found = True
                        break

                    # ✅ 单目标微调
                    target = det[0]
                    x1, y1, x2, y2 = map(int, target[:4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    dx = cx - frame.shape[1] // 2
                    dy = cy - frame.shape[0] // 2

                    scale_x = dx * 0.004
                    scale_y = dy * 0.004
                    adjust_north = -scale_y
                    adjust_east  = scale_x

                    new_x = cur_x + adjust_north
                    new_y = cur_y + adjust_east

                    drone_fly_with_camera(new_x, new_y, base_height, 1)
                    print(f"1调整至 {new_x}, {new_y}, {base_height}")

                    found = True
                    break
                drone_hover(1.0 / 30)

            if not found:
                print("目标未识别，回退重新搜索")
                flag = 1
                continue

            if base_height > -1.1:
                print("已接近地面，转为OpenCV圆心对准")
                target_stage = -1
            else:
                target_stage += 1
            drone_hover(1.0)

        # ---------------- OpenCV 圆心对准 + YOLO辅助 ----------------
        elif target_stage == -1:
            base_height = -1.0
            frame = get_current_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                        param1=100, param2=30, minRadius=10, maxRadius=100)

            # YOLO识别
            img0 = frame.copy()
            img = letterbox(img0, new_shape=640, stride=32, auto=True)[0]
            img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.half() if device.type != 'cpu' else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            if model.half:  # 只有模型使用 half 才转换
                img_tensor = img_tensor.half()

            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            yolo_box = None
            if pred and len(pred[0]):
                det = pred[0]
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                yolo_box = det[0][:4].int().tolist()
                x1, y1, x2, y2 = yolo_box
                yolo_cx = (x1 + x2) // 2
                yolo_cy = (y1 + y2) // 2

            if circles is not None:
                circles = np.uint16(np.around(circles))
                cx, cy, r = circles[0][0]

                if yolo_box and not (x1 < cx < x2 and y1 < cy < y2):
                    print("圆心未落在YOLO框中，使用YOLO中心点进行微调")
                    cx, cy = yolo_cx, yolo_cy  # 替换为YOLO中心点

                dx = cx - frame.shape[1] // 2
                dy = cy - frame.shape[0] // 2
                print(f"对准误差 dx: {dx}, dy: {dy}")

                if abs(dx) < 20 and abs(dy) < 20:
                    print("对准成功，悬停10s，进入下一阶段")
                    drone_hover(10)
                    target_stage = 0
                    init34 = 0
                    flag = 4 if flag == 3 else 5
                else:
                    cur_x, cur_y, _ = mav.uavPosNED
                    scale_x = dx * 0.002
                    scale_y = dy * 0.002
                    adjust_north = -scale_y
                    adjust_east = scale_x
                    new_x = cur_x + adjust_north
                    new_y = cur_y + adjust_east
                    drone_fly_with_camera(new_x, new_y, base_height, 0.7)
                    print(f"微调至 {new_x}, {new_y}, {base_height}")
            else:
                print("未检测到圆心，重试中")



    if flag == 5:
        print("[Flag 5] 返回搜索区域")
        drone_fly_with_camera(*search_position, 5)
        flag = 6

    elif flag == 6:
        print("[Flag 6] 前往悬停点")
        drone_fly_with_camera(*hover_position, 20)
        drone_hover(10)
        flag = 7

    elif flag == 7:
        print("[Flag 7] 返回原点准备降落")
        drone_fly_with_camera(*home_position, 40)
        mav.sendMavLand(0, 0, 0)
        print("降落指令已发送，任务完成")
        flag = 8  # 终止或等待

    # mav.SendPosFRD
    # mav.SendPosNED
    # mav.SendPosGlobal