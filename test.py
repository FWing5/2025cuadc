import cv2
from yololite_detector import YOLOv5LiteDetector

detector = YOLOv5LiteDetector(weights='YOLOv5-Lite\\yolo_lite_cuadc8\\weights\\best.pt',view_img=True)
img = cv2.imread('2.jpg')
detections, result_img = detector.detect(img)

print(detections)
cv2.imwrite('result.jpg', result_img)