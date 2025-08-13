from drone_controller import DroneController
from yolo_detector import YOLOv5Detector
from vision_utils import *

def process_frame_for_targets(
    frame,
    detector,
    real_quad = [
                    (38, -4),  # 图像左上 → 世界右上
                    (38, 4),   # 图像右上 → 世界右下
                    (32, 4),   # 图像右下 → 世界左下
                    (32, -4),  # 图像左下 → 世界左上
                ],
    expect_target_num=3,
    draw_quad=True,
):

    """
    在单帧中完成：
      1) 找最大四边形
      2) YOLO 识别四边形内目标
      3) 若目标数满足 expect_target_num，计算像素->世界的单应变换，并输出世界坐标
    返回一个 dict，包含 success、原因、四边形、单应矩阵、目标像素/世界坐标、调试图像等。
    """
    result = {
        "success": False,
        "reason": "",
        "quad": None,
        "perspective_matrix": None,
        "targets_pixel": [],
        "targets_world": [],
        "debug_img": None,
    }

    # 1) 输入检查
    if frame is None:
        result["reason"] = "frame is None"
        return result
    if detector is None:
        result["reason"] = "detector is None"
        return result
    if not isinstance(real_quad, (list, tuple)) or len(real_quad) != 4:
        result["reason"] = "real_quad must be 4 points"
        return result

    # 2) 查找最大四边形
    quad_found = find_largest_quadrilateral(frame)
    if quad_found is None:
        result["reason"] = "no quadrilateral found"
        return result

    # 假设 quad_found 是 4 点
    quad = quad_found
    # 3) YOLO 检测并筛选四边形内目标
    targets_inside, debug_img = yolo_detection_on_quad(frame, quad, detector)
    result["debug_img"] = debug_img

    if targets_inside is None:
        result["reason"] = "yolo_detection_on_quad returned None"
        return result

    if len(targets_inside) != expect_target_num:
        result["reason"] = f"targets count {len(targets_inside)} != {expect_target_num}"
        # 调试图像上可画出四边形
        if draw_quad and debug_img is not None:
            import cv2
            for i in range(4):
                cv2.line(debug_img, tuple(map(int, quad[i])), tuple(map(int, quad[(i + 1) % 4])), (255, 0, 0), 2)
        return result

    # 4) 对目标按 cx 从左到右排序，方便后续逻辑
    targets_inside = sorted(targets_inside, key=lambda x: x[0][0])
    result["targets_pixel"] = targets_inside

    # 5) 对 quad 做一致化排序（与你下游约定一致）
    quad_sorted = sort_quad_points(quad)
    result["quad"] = quad_sorted

    # 6) 计算像素->世界的单应矩阵
    perspective_matrix = compute_perspective_transform(quad_sorted, real_quad)
    result["perspective_matrix"] = perspective_matrix

    # 7) 将目标像素坐标变换到世界坐标
    targets_world = []
    for (cx, cy), cls in targets_inside:
        wx, wy = pixel_to_world(cx, cy, perspective_matrix)
        targets_world.append((wx, wy))
    result["targets_world"] = targets_world

    # 8) 在调试图像上画出四边形边界（可选）
    if draw_quad and debug_img is not None:
        import cv2
        for i in range(4):
            cv2.line(debug_img, tuple(map(int, quad_sorted[i])), tuple(map(int, quad_sorted[(i + 1) % 4])), (255, 0, 0), 2)

    result["success"] = True
    return result