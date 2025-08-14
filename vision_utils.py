import cv2
import math
import numpy as np
import requests
import socket
import sys
import time

def find_largest_quadrilateral(frame):
    """
    在图像中查找面积最大的四边形轮廓，且面积必须大于图像总面积的 50%。

    参数：
        frame (ndarray): 输入的BGR图像帧。

    返回：
        ndarray 或 None: 4个顶点坐标的二维数组（4x2），如果未找到符合条件的四边形则返回 None。
    """
    height, width = frame.shape[:2]
    frame_area = width * height  # 计算图像总面积

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # 添加面积限制条件
            if area > max_area and area > 0.5 * frame_area:
                max_area = area
                best_quad = approx

    if best_quad is not None:
        return best_quad.reshape(4, 2)
    return None

def sort_quad_points(quad):
    """
    将四边形顶点按顺时针顺序排序：左上，右上，右下，左下。（对齐图像坐标与现实坐标）

    参数：
        quad (ndarray): 形状为 (4, 2) 的四边形顶点数组。

    返回：
        ndarray: 按顺时针顺序排列的顶点。
    """
    # pts: 4x2 array
    # 返回顺时针排序：TL, TR, BR, BL
    pts = np.array(quad, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    return pts[sorted_idx]

def compute_perspective_transform(src_points, dst_points):
    """
    计算透视变换矩阵。

    参数：
        src_points (ndarray): 源图像中的4个点。
        dst_points (ndarray): 目标图像中的4个点。

    返回：
        ndarray: 3x3的透视变换矩阵。
    """
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    return matrix

def pixel_to_world(x, y, matrix):
    """
    将图像像素坐标通过透视变换转换为世界坐标。

    参数：
        x (float): 图像中的x坐标。
        y (float): 图像中的y坐标。
        matrix (ndarray): 3x3的透视变换矩阵。

    返回：
        tuple: (x, y) 在目标坐标系下的位置。
    """
    pt = np.array([[[x, y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(pt, matrix)
    return transformed[0][0][0], transformed[0][0][1]

def point_in_quad(point, quad):
    """
    判断一个点是否在给定的四边形内。

    参数：
        point (tuple): 点的坐标 (x, y)。
        quad (ndarray): 四边形的四个顶点坐标。

    返回：
        bool: 如果点在四边形内部或边上，则为 True；否则为 False。
    """
    poly = np.array(quad, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

def detect_circles(frame):
    """
    在图像中检测圆形。

    参数：
        frame (ndarray): 输入的BGR图像。

    返回：
        ndarray 或 None: 每个圆以 (x, y, r) 形式返回的数组，表示圆心和半径。如果未检测到圆，则返回 None。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                            param1=100, param2=30, minRadius=10, maxRadius=100)

def get_closest_center_offset(detections, img_width, img_height):
    """
    输入检测结果和图像尺寸，返回最靠近图像中心的目标中心与图像中心的偏移量(dx, dy)。

    参数：
        detections: list of dict，每个包含 'bbox' = [x1,y1,x2,y2]
        img_width: int，图像宽度
        img_height: int，图像高度

    返回：
        (dx, dy): float，目标中心与图像中心的像素偏移
        如果没有检测目标，返回 None
    """

    if not detections:
        return None

    cx_img = img_width / 2
    cy_img = img_height / 2

    min_dist = None
    closest_dx = None
    closest_dy = None

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        dx = cx - cx_img
        dy = cy - cy_img

        dist = math.hypot(dx, dy)
        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
            closest_dx = dx
            closest_dy = dy

    return closest_dx, closest_dy


def get_circle_offset_in_closest_bbox(detections, circles, img_width, img_height):
    """
    在最近的检测框中找到最近的圆心，返回画面中心与该圆心的 dx、dy

    参数：
        detections: list of dict，每个包含 'bbox' = [x1,y1,x2,y2]
        circles: ndarray 或 None，来自 detect_circles 的结果 (N, 1, 3) 或 (N, 3)
        img_width: int，图像宽度
        img_height: int，图像高度

    返回：
        (dx, dy): float，画面中心与圆心的像素偏移
        如果没有检测框或没有圆，则返回 None
    """
    if not detections or circles is None:
        return None

    # 找出离画面中心最近的检测框
    cx_img = img_width / 2
    cy_img = img_height / 2
    min_dist_box = None
    closest_box = None

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        dist_box = math.hypot(cx_box - cx_img, cy_box - cy_img)
        if (min_dist_box is None) or (dist_box < min_dist_box):
            min_dist_box = dist_box
            closest_box = (x1, y1, x2, y2)

    if closest_box is None:
        return None

    # 过滤在该检测框内的圆
    x1, y1, x2, y2 = closest_box
    circles = np.squeeze(circles)  # 可能是 (N,1,3) 或 (N,3)
    if circles.ndim == 1:
        circles = np.expand_dims(circles, axis=0)

    circles_in_box = []
    for (cx, cy, r) in circles:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            circles_in_box.append((cx, cy, r))

    if not circles_in_box:
        return None

    # 找出该框内离画面中心最近的圆
    min_dist_circle = None
    closest_circle = None
    for (cx, cy, r) in circles_in_box:
        dist_circle = math.hypot(cx - cx_img, cy - cy_img)
        if (min_dist_circle is None) or (dist_circle < min_dist_circle):
            min_dist_circle = dist_circle
            closest_circle = (cx, cy)

    if closest_circle is None:
        return None

    dx = closest_circle[0] - cx_img
    dy = closest_circle[1] - cy_img
    return dx, dy

def send_frame(frame, timeout_sec=2.0):
    """
    frame: numpy BGR图像
    timeout_sec: 网络超时时间（秒）
    """
    url = "http://192.168.4.6:8000/upload_frame"
    try:
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            print("编码失败")
            return False
        files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
        r = requests.post(url, files=files, timeout=timeout_sec)
        return r.status_code == 200
    except requests.exceptions.Timeout:
        print(f"[WARN] 发送超时({timeout_sec}s)，已跳过")
        return False
    except Exception as e:
        print(f"[ERROR] 发送失败: {e}")
        return False


def yolo_detection_on_quad(frame, quad, detector):
    """
    在给定四边形内，使用YOLO检测目标并筛选在四边形内部的目标。

    返回：
        targets_inside: list of ((cx, cy), class_id)
        img_with_markers: 带检测框和圆点的调试图像
    """
    img = cv2.resize(frame, (640, 640))

    # 这里调用你的检测模块接口
    detections, result_img = detector.detect(frame)  

    targets_inside = []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if point_in_quad((cx, cy), quad):
            targets_inside.append(((cx, cy), det['class_id']))
            cv2.circle(result_img, (cx, cy), 5, (0, 255, 0), -1)
        else:
            cv2.circle(result_img, (cx, cy), 5, (0, 0, 255), -1)

    return targets_inside, result_img

class Camera:
    def __init__(self, width=640, height=480, mode = "real", index=0):
        if mode == "real":
            self.mode = "real"
            self.cap = cv2.VideoCapture(index)

            if not self.cap.isOpened():
                raise RuntimeError("无法连接到摄像头。")

            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self.width = width
            self.height = height
        else:
            self.mode = "simu"
            # 控制接口与图像采集接口
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
            self.vis = vis

    def read(self):
        """读取一帧图像"""
        if self.mode == "real":
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("无法读取摄像头画面。")
            return frame
        elif self.mode == "simu":
            if self.vis.hasData[0]:
                frame = self.vis.Img[0]
                if frame is not None and frame.size > 0:
                    return frame
            return None

    def show(self, window_name="Camera"):
        """显示实时画面（按 q 退出）"""
        while True:
            frame = self.read()
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()

    def release(self):
        """释放摄像头资源"""
        self.cap.release()
        cv2.destroyAllWindows()