import time
import select
import sys
from drone_controller import DroneController
from yolo_detector import YOLOv5Detector
from vision_utils import *

import time
import logging
import math
import traceback

# 假设你已有以下类
# from your_pkg import DroneController, YOLOv5Detector, Camera

# 基础配置
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 安全范围（按需要调整）
MAX_HORIZONTAL_RADIUS = 100.0  # m
MIN_Z = -10.0  # 最高飞到 10m（z=-10）
MAX_Z = 0.0    # 地面高度 z=0（NED）
DEFAULT_WAIT = 0.05

def try_send_frame_from_cam(cam):
    try:
        frame = cam.read()
        try:
            send_frame(frame)  # 如果你工程里有 send_frame，就会被调用
        except NameError:
            pass  # 没有 send_frame 就跳过
        except Exception as e:
            logging.warning(f"send_frame 发送失败: {e}")
    except Exception as e:
        logging.warning(f"相机读取失败: {e}")

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def within_bounds(x, y, z):
    # 简单边界检查（可改成你的地理围栏逻辑）
    if math.sqrt(x*x + y*y) > MAX_HORIZONTAL_RADIUS:
        return False
    if z < MIN_Z or z > MAX_Z:
        return False
    return True

def safe_call(step_name, func, on_error_continue=True):
    try:
        logging.info(f"[STEP] {step_name} 开始")
        func()
        logging.info(f"[STEP] {step_name} 完成")
    except Exception as e:
        logging.error(f"[STEP] {step_name} 异常: {e}")
        traceback.print_exc()
        if not on_error_continue:
            raise
        else:
            logging.warning(f"[STEP] {step_name} 发生异常，按要求跳过继续后续步骤")

def safe_move_to(drone, x, y, z, duration=5.0, step_label="", check_bounds=True):
    # 安全的飞到某个局部 NED 绝对点（假设 fly_to 为绝对目标）
    # 如果你的 fly_to 是相对机体系位移，请在此处改为相对移动逻辑
    if check_bounds and not within_bounds(x, y, z):
        raise ValueError(f"目标点越界: ({x}, {y}, {z})")

    logging.info(f"[MOVE] {step_label} -> 目标({x:.2f}, {y:.2f}, {z:.2f}) 持续 {duration}s")
    # 加入简易重试机制
    attempts = 2
    for i in range(attempts):
        try:
            drone.fly_to(x, y, z, duration)
            return
        except Exception as e:
            logging.warning(f"[MOVE] fly_to 异常（第 {i+1}/{attempts} 次）: {e}")
            if i == attempts - 1:
                raise
            time.sleep(0.5)

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

def detect_quad_phase(detector, camera, drone):
    start_time = time.time()
    while True and time.time() - start_time < 15:
        drone.hover(0.05)
        try:
            frame = camera.read()
            result = process_frame_for_targets(frame, detector)
            
            if result.get("success") if isinstance(result, dict) else None:
                return result
        except Exception as e:
            logging.warning(f"[WARMUP] 检测方框时异常: {e}")

def deliver_bottle_phase(
    drone,
    cam,
    detector,
    bottle,
    result,
    target_idx = None,  # 新增参数：指定 targets_world 的索引
    # 参数配置
    loop_hz=0.5,
    target_offset_threshold=30.0,
    cam_width=640,
    cam_height=480,
    circle_stage_altitude=1.6,
    no_detect_descend_interval=10.0,
    descend_step=0.5,
    fly_target_altitude=-2.5,
    fly_duration=5.0,
    allow_key_abort=True,
):
    """
    返回: (success: bool, reason: str)
    """

    import sys, time, math

    def _select_nearest_world_target(targets_world, cur_xy):
        if not targets_world:
            return None
        cx, cy = cur_xy
        best = None
        best_d = float("inf")
        for (wx, wy) in targets_world:
            d = (wx - cx) ** 2 + (wy - cy) ** 2
            if d < best_d:
                best_d = d
                best = (wx, wy)
        return best

    # 1) 若有 targets_world，先飞过去
    try:
        targets_world = result.get("targets_world") if isinstance(result, dict) else None
    except Exception:
        targets_world = None

    try:
        cur_x, cur_y, cur_z = drone.get_position()
    except Exception:
        cur_x, cur_y, cur_z = 0.0, 0.0, 999.0

    target_xy = None

    if targets_world:
        # 优先使用指定索引
        if target_idx is not None:
            if 0 <= target_idx < len(targets_world):
                target_xy = targets_world[target_idx]
                print(f"[INFO] 使用指定索引 {target_idx} 的世界坐标: {target_xy}")
            else:
                print(f"[WARN] 指定的 target_idx {target_idx} 超出范围；将使用最近目标策略")
                target_xy = _select_nearest_world_target(targets_world, (cur_x, cur_y))
        else:
            # 没有指定索引，使用最近目标策略
            target_xy = _select_nearest_world_target(targets_world, (cur_x, cur_y))

        if target_xy is not None:
            wx, wy = target_xy
            print(f"[INFO] 飞往世界坐标: ({wx:.2f}, {wy:.2f}, {fly_target_altitude:.2f})")
            try:
                drone.fly_to(wx, wy, fly_target_altitude, fly_duration)
            except Exception as e:
                print(f"[WARN] fly_to 发生异常: {e}; 将直接进入视觉搜索")
        else:
            print("[INFO] targets_world 非空，但未选出目标；直接进入视觉搜索")
    else:
        print("[INFO] targets_world 为空；直接进入视觉搜索")

    # 2) 进入视觉引导 + 投放主循环
    last_iter_time = 0.0
    last_detect_time = time.time()
    start_time = time.time()

    while True and (time.time()-start_time) < 150:

        # 读帧
        try:
            frame = cam.read()
            if frame is None:
                print("[WARN] 无法读取相机帧")
                continue
        except Exception as e:
            print(f"[ERROR] 相机读取异常: {e}")
            return False, "camera_error"

        # 控制循环频率
        now = time.time()
        if now - last_iter_time < 1.0 / loop_hz:
            continue
        last_iter_time = now

        # 目标检测
        try:
            dtc, img = detector.detect(frame)
        except Exception as e:
            print(f"[ERROR] 检测器异常: {e}")
            return False, "detector_error"

        if send_frame is not None:
            try:
                send_frame(img if img is not None else frame)
            except Exception:
                pass

        if not dtc:
            # 没有检测到任何目标
            print("[WARN] 未检测到目标")
            # 超过 no_detect_descend_interval 未检测到目标，则上升
            cur_x, cur_y, cur_z = drone.get_position()
            if (now - last_detect_time >= no_detect_descend_interval) and (cur_z > -3.5):
                try:
                    print(f"[ACTION] {no_detect_descend_interval:.0f}s 无目标，上升 {descend_step}m")
                    drone.down(-descend_step, 1)
                except Exception as e:
                    print(f"[WARN] 上升失败: {e}")
                last_detect_time = now
            continue

        # 有目标，刷新“最近一次检测到目标”的时间
        last_detect_time = now
        try:
            _, _, cur_z = drone.get_position()
        except Exception:
            # 若此处取高失败，假定仍需继续常规对准
            cur_z = 999.0

        print(f"[DEBUG] 检测到 {len(dtc)} 个目标, 当前高度: {cur_z:.2f} m")

        circle_stage_times = 0
        if cur_z >= -circle_stage_altitude:

            circle_stage_times += 1
            if circle_stage_times > 10:
                print(f"[INFO] 超过尝试次数，直接投放bottle={bottle}")
                try:
                    drone.drop_bottle(bottle)
                except Exception as e:
                    print(f"[ERROR] 投放失败: {e}")
                    return False, "drop_failed"
                return True, "dropped"

            # 3) 进入圆形检测精对准阶段
            # print("[INFO] 进入圆形检测阶段...")
            try:
                circles = detect_circles(frame)
            except Exception as e:
                print(f"[WARN] detect_circles 异常: {e}")
                circles = None

            if circles is not None:
                try:
                    offset = get_circle_offset_in_closest_bbox(dtc, circles, cam_width, cam_height)
                except Exception as e:
                    print(f"[WARN] get_circle_offset_in_closest_bbox 异常: {e}")
                    offset = None

                if offset:
                    dx, dy = offset
                    print(f"[DEBUG] （圆形辅助检测阶段）圆心偏移 dx={dx:.2f}, dy={dy:.2f}")
                    if abs(dx) < target_offset_threshold*2 and abs(dy) < target_offset_threshold*2:
                        print(f"[ACTION] 投放物品: bottle={bottle}")
                        try:
                            drone.drop_bottle(bottle)
                        except Exception as e:
                            print(f"[ERROR] 投放失败: {e}")
                            return False, "drop_failed"
                        return True, "dropped"
                    else:
                        print("[ACTION] 调整位置以对准圆心")
                        try:
                            drone.adjust_position_by_pixel_offset(dx, dy, scale=0.001)
                        except Exception as e:
                            print(f"[WARN] 调整位置失败: {e}")
                else:
                    print("[WARN] 未能获得圆心偏移")
            else:
                print("[WARN] 未检测到圆形")

        else:
            # 4) 常规阶段：以最近目标的框中心对准；对准后下降 0.5m
            try:
                dx, dy = get_closest_center_offset(dtc, cam_width, cam_height)
            except Exception as e:
                print(f"[WARN] get_closest_center_offset 异常: {e}")
                dx, dy = None, None

            if dx is None or dy is None:
                print("[WARN] 无法计算目标中心偏移")
                continue

            print(f"[DEBUG] 目标中心偏移 dx={dx:.2f}, dy={dy:.2f}")
            if abs(dx) < target_offset_threshold and abs(dy) < target_offset_threshold:
                print(f"[ACTION] 对准完成，下降 {descend_step}m")
                try:
                    drone.down(descend_step, 1)
                except Exception as e:
                    print(f"[WARN] 下降失败: {e}")
            else:
                print("[ACTION] 调整位置以对准目标框中心")
                try:
                    drone.adjust_position_by_pixel_offset(dx, dy)
                except Exception as e:
                    print(f"[WARN] 调整位置失败: {e}")

def main():
    try:
        # 基础初始化
        logging.info("[INIT] 初始化检测器/相机")
        detector = YOLOv5Detector(view_img=False)
        try:
            cam = Camera(CAM_WIDTH, CAM_HEIGHT, index=0)
        except Exception as e0:
            print(f"Camera index 0 failed: {e0}")
            try:
                cam = Camera(CAM_WIDTH, CAM_HEIGHT, index=1)
                print("Fallback to camera index 1.")
            except Exception as e1:
                raise RuntimeError(f"Both camera 0 and 1 failed. err0={e0}, err1={e1}")

        # 启动前拍一帧
        try_send_frame_from_cam(cam)

        # 预热检测器（也可带上 send_frame）
        def warmup():
            logging.info("[INFO] 正在预热检测器...")
            for _ in range(5):
                try:
                    frame = cam.read()
                    _ = detector.detect(frame)
                except Exception as e:
                    logging.warning(f"[WARMUP] 预热帧异常: {e}")
                time.sleep(0.05)
            logging.info("[INFO] 检测器预热完成")
        safe_call("预热检测器", warmup, on_error_continue=True)

        input("[INFO] 按回车键开始初始化无人机...")
        try:
            drone = DroneController()
        except Exception as e:
            logging.warning(f"[WARMUP] 无人机初始化异常: {e}")
            sys.exit(1)

        # 起飞到 2m 高（z=-2）
        # 注：若你的飞控需要先从地面起飞，请确保在此之前已起飞或允许位置控制起飞
        safe_call("起飞到2m(z=-2)", lambda: safe_move_to(drone, 0.0, 0.0, -2.0, duration=8.0, step_label="起飞到2m"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        # 飞至 x=34.5 y=0 z=-2
        safe_call("飞至(34.5,0,-2)", lambda: safe_move_to(drone, 34.5, 0.0, -2.0, duration=15.0, step_label="前往34.5m,2m高"), on_error_continue=True)
        # try_send_frame_from_cam(cam)

        safe_call("飞至(34.5,0,-4.5)", lambda: safe_move_to(drone, 34.5, 0.0, -4.5, duration=4.0, step_label="前往34.5m,4.5m高"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        # 阶段
        quad_result = detect_quad_phase(detector, cam, drone)

        safe_call("试图投放第一个瓶子", lambda: deliver_bottle_phase(drone, cam, detector, "front", quad_result, 1), on_error_continue=True)

        safe_call("试图投放第二个瓶子", lambda: deliver_bottle_phase(drone, cam, detector, "back", quad_result, 2), on_error_continue=True)


    
        # 飞至 x=34.5 y=0 z=-2
        safe_call("返回(34.5,0,-2)", lambda: safe_move_to(drone, 34.5, 0.0, -2.0, duration=10.0, step_label="回到2m高"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        # 飞至 x=59.5 y=0 z=-2
        safe_call("飞至(59.5,0,-2)", lambda: safe_move_to(drone, 59.5, 0.0, -2.0, duration=20.0, step_label="前往59.5m,2m高"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        safe_call("飞至(59.5,-1.5,-2)", lambda: safe_move_to(drone, 59.5, -1.5, -2.0, duration=8.0, step_label="左1.5m"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        safe_call("飞至(59.5,1.5,-2)", lambda: safe_move_to(drone, 59.5, 1.5, -2.0, duration=10.0, step_label="右1.5m"), on_error_continue=True)
        try_send_frame_from_cam(cam)
        time.sleep(3)

        # 飞至 x=59.5 y=0 z=-2
        safe_call("返回(59.5,0,-2)", lambda: safe_move_to(drone, 59.5, 0.0, -2.0, duration=8.0, step_label="回到2m高(59.5,0)"), on_error_continue=True)
        try_send_frame_from_cam(cam)

        # 返回 0 0 -2
        safe_call("返回(0,0,-2)", lambda: safe_move_to(drone, 0.0, 0.0, -2.0, duration=40.0, step_label="返航2m高"), on_error_continue=True)
        # try_send_frame_from_cam(cam)

        # 降落
        def land_and_shutdown():
            logging.info("[INFO] 返回起始点并降落")
            try:
                drone.land()
            except Exception as e:
                logging.error(f"[LAND] 降落指令异常: {e}")
            time.sleep(10)
            try:
                drone.shutdown()
            except Exception as e:
                logging.error(f"[SHUTDOWN] 关机异常: {e}")
            logging.info("[INFO] 程序结束")
            
        safe_call("降落与关机", land_and_shutdown, on_error_continue=True)
        
    except KeyboardInterrupt:
        logging.info("退出中")
        try:
            drone.land()
        except Exception as e:
            logging.error(f"[LAND] 降落指令异常: {e}")
        time.sleep(10)
        try:
            drone.shutdown()
            sys.exit(1)
        except Exception as e:
            logging.error(f"[SHUTDOWN] 关机异常: {e}")
        logging.info("[INFO] 程序结束")
    finally:
        sys.exit(1)


if __name__ == '__main__':
    main()