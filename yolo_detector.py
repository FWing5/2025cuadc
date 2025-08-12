import cv2
import torch
import sys
import platform
import pathlib
from pathlib import Path

# 兼容 Windows 权重路径
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# 导入 YOLOv5 模型
sys.path.insert(0, './myyolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import plot_one_box
from utils.datasets import letterbox


class YOLOv5Detector:
    def __init__(self,
                 weights='myyolov5/weights/best.pt',
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 device='',
                 view_img=False,
                 use_letterbox=True):
        """
        YOLOv5 推理类
        """
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.view_img = view_img
        self.use_letterbox = use_letterbox

    def detect(self, img0):
        """
        单张图像推理接口
        img0: numpy array (BGR) 原始图像

        返回: (detections, result_img)
            detections: list of dict, 每个dict包含bbox, conf, class_id, label
            result_img: numpy array, 已画框的图像
        """
        # 图像预处理
        if self.use_letterbox:
            img = letterbox(img0, new_shape=self.img_size, stride=self.stride, auto=True)[0]
        else:
            img = cv2.resize(img0, (self.img_size, self.img_size))

        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 推理
        pred = self.model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        detections = []
        im0 = img0.copy()

        if pred and len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                bbox = [int(x.item()) for x in xyxy]
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(bbox, im0, label=label, color=(0, 255, 0), line_thickness=2)
                detections.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'conf': conf.item(),
                    'class_id': int(cls),
                    'label': self.names[int(cls)]
                })

        if self.view_img:
            cv2.imshow('YOLOv5 Detection', im0)
            cv2.waitKey(1)

        return detections, im0


if __name__ == '__main__':
    # 示例：实时摄像头推理
    cap = cv2.VideoCapture(0)
    # 切换 use_letterbox=True/False 对比速度
    detector = YOLOv5Detector(view_img=True, use_letterbox=True)

    import time
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        detections, result_img = detector.detect(frame)
        print(f"推理耗时: {(time.time()-start)*1000:.2f} ms, 检测结果数: {len(detections)}")
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
