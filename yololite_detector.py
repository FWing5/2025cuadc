import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, './YOLOv5-Lite')

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

class YOLOv5LiteDetector:
    def __init__(self, 
                 weights='weights/best.pt', 
                 img_size=640, 
                 conf_thres=0.45, 
                 iou_thres=0.5,
                 device='',
                 view_img=False):
        
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.img_size = check_img_size(img_size, s=int(self.model.stride.max()))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.view_img = view_img

    def detect(self, img0):
        """
        单张图像推理接口
        img0: numpy array (BGR) 原始图像

        返回: (detections, result_img)
            detections: list of dict, 每个dict包含bbox, conf, class_id, label
            result_img: numpy array, 已画框的图像
        """
        # Resize and pad image to model input size
        from utils.datasets import letterbox
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert BGR to RGB, transpose to channel first, and convert to torch tensor
        img = img[:, :, ::-1].copy().transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0-255 to 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        detections = []
        im0 = img0.copy()

        if pred is not None and len(pred):
            # 根据 letterbox 缩放恢复原图坐标
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=2)
                detections.append({
                    'bbox': [int(x.item()) for x in xyxy],  # [x1, y1, x2, y2]
                    'conf': conf.item(),
                    'class_id': int(cls),
                    'label': self.names[int(cls)]
                })

        if self.view_img:
            cv2.imshow('YOLOv5Lite Detection', im0)
            cv2.waitKey(1)

        return detections, im0