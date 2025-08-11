import cv2
import numpy as np
import requests

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
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
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

def send_frame(frame, url="http://0.0.0.0:8000/upload_frame"):
    """
    frame: numpy BGR图像
    """
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        print("编码失败")
        return False
    files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    r = requests.post(url, files=files)
    return r.status_code == 200

class Camera:
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("无法连接到摄像头。")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height

    def read(self):
        """读取一帧图像"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头画面。")
        return frame

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
