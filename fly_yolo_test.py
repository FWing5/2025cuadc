import time
from drone_controller import DroneController
from vision_utils import Camera
from yololite_detector import YOLOv5LiteDetector

"""
==========备注===========
frame: 未处理摄像头视频帧
img: YOLO处理后的帧
"""
CAM_WIDTH = 640
CAM_HEIGHT = 480

drone = DroneController()
detector = YOLOv5LiteDetector(view_img=False)
cam = Camera(CAM_WIDTH, CAM_HEIGHT)
send_frame(cam.read())

drone.fly_to(0, 0, -4, 10)
send_frame(cam.read())

# 预热
for _ in range(5):
    frame = cam.read()
    _ = detector.detect(frame)

while True:
    frame = cam.read()
    dtc, img = detector.detect(frame)
    if dtc == None:
        print("未检测到目标")
    dx, dy = get_closest_center_offset(dtc, CAM_WIDTH, CAM_HEIGHT)
    if





drone.fly_to(0, 0, -4, 10)

drone.land()

time.sleep(10)
drone.shutdown()