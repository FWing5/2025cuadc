import time
import math
import PX4MavCtrlV4 as PX4MavCtrl

class DroneController:
    def __init__(self, platform="raspi-usb"):
        """
        初始化无人机连接，并进入Offboard模式与解锁。

        参数：
            platform (str): 连接平台类型，可选值：
                - "raspi-usb": 树莓派USB连接
                - "raspi-com": 树莓派串口连接
                - "windows": Windows设备连接
                - "simu": 使用仿真环境
        """
        platform = platform.lower()

        try:
            # Step 1: 选择连接方式
            if platform == "raspi-usb":
                print("[INFO] 使用树莓派 USB 连接")
                self.mav = PX4MavCtrl.PX4MavCtrler(1, Com='/dev/ttyACM0')

            elif platform == "raspi-com":
                print("[INFO] 使用树莓派 串口连接")
                self.mav = PX4MavCtrl.PX4MavCtrler(1, Com='/dev/ttyAMA0')

            elif platform == "windows":
                print("[INFO] 使用 Windows 串口连接")
                self.mav = PX4MavCtrl.PX4MavCtrler(1, Com='COM3')
            
            elif platform == "simu":
                print("[INFO] 使用 仿真")
                self.mav = PX4MavCtrl.PX4MavCtrler(1)

            else:
                raise ValueError(f"[ERROR] 不支持的平台类型: '{platform}'，请使用 'raspi-usb'、'raspi-com' 或 'windows'。")

            # Step 2: 初始化 MAVLink 循环
            print("[INFO] 初始化 MAVLink 通信循环...")
            self.mav.InitMavLoop()
            time.sleep(1)

            # Step 3: 进入 Offboard 模式
            print("[INFO] 进入 Offboard 模式...")
            self.mav.initOffboard()
            time.sleep(1)

            # Step 4: 解锁无人机（Arm）
            print("[INFO] 解锁无人机 (Arm)...")
            self.mav.SendMavArm(True)
            time.sleep(1)

            # Step 5: 获取初始 yaw
            self.initial_yaw = self.mav.uavAngEular[2]
            print(f"[INFO] 初始 yaw: {math.degrees(self.initial_yaw):.2f}°")


            print("[SUCCESS] 无人机已连接并准备就绪！")

        except Exception as e:
            print(f"[ERROR] 初始化无人机通信失败: {e}")
            self.mav = None  # 防止self.mav未定义引发后续错误

    def shutdown(self):
        print("[INFO] 上锁无人机（Disarm）...")
        self.mav.SendMavArm(False)
        time.sleep(1)

        print("[INFO] 停止 Offboard 模式...")
        self.mav.endOffboard()
        time.sleep(1)

        print("[INFO] 停止 MAVLink 通信循环...")
        self.mav.endMavLoop()
        time.sleep(1)

        print("[SUCCESS] 无人机已终止。")

    def get_position(self):
        return self.mav.uavPosNED
    
    def get_yaw(self):
        return self.mav.uavAngEular[2]

    def fly_to(self, x_body, y_body, z_body, duration=2):
        """
        机体系（前、右、下）坐标飞行
        duration 秒
        """
        print(f"前 {x_body} 米，右 {y_body} 米，高度{-z_body}米，执行 {duration} 秒")
        # 旋转矩阵：机体 → 全局 NED
        x_north = x_body * math.cos(self.initial_yaw) - y_body * math.sin(self.initial_yaw)
        y_east  = x_body * math.sin(self.initial_yaw) + y_body * math.cos(self.initial_yaw)
        z_down  = z_body  # 下方向在 NED 中不变

        start_time = time.time()
        while time.time() - start_time < duration:
            self.mav.SendPosNED(x_north, y_east, z_down, 0)
            time.sleep(0.05)  # 控制频率 20Hz
    
    def fly_toward(self, x_body_offset, y_body_offset, z_body_offset, duration=2):
        """
        相对当前位置的机体偏移飞行，传入相对偏移，内部计算目标全局坐标并发送
        """
        print(f"相对当前位置偏移飞行：前 {x_body_offset} 米，右 {y_body_offset} 米，执行 {duration} 秒")

        cur_x, cur_y, cur_z = self.get_position()

        # 机体坐标系偏移转换到全局NED偏移
        offset_x = x_body_offset * math.cos(yaw_relative) - y_body_offset * math.sin(yaw_relative)
        offset_y = x_body_offset * math.sin(yaw_relative) + y_body_offset * math.cos(yaw_relative)
        offset_z = z_body_offset

        # 新目标全局坐标 = 当前全局坐标 + 偏移
        target_x = cur_x + offset_x
        target_y = cur_y + offset_y
        target_z = cur_z + offset_z

        start_time = time.time()
        while time.time() - start_time < duration:
            self.mav.SendPosNED(target_x, target_y, target_z, 0)
            time.sleep(0.05)

    def hover(self, duration=2):
        print(f"悬停 {duration} 秒")
        start_time = time.time()
        cur_x, cur_y, cur_z = self.get_position()
        while time.time() - start_time < duration:
            self.mav.SendPosNED(cur_x, cur_y, cur_z, 0)
            time.sleep(0.05)  # 控制频率，200ms发送一次

    def down(self, d, duration=2):
        print(f"原地下降 {d} 米，执行 {duration} 秒")
        start_time = time.time()
        cur_x, cur_y, cur_z = self.get_position()
        while time.time() - start_time < duration:
            self.mav.SendPosNED(cur_x, cur_y, cur_z + d, 0)
            time.sleep(0.05)  # 控制频率，200ms发送一次

    def land(self, x = 0, y = 0, z = 0):
        print(f"降落至: ({x}, {y}, {z}) ，降落成功后会自动disarm")
        self.mav.sendMavLand(0, 0, 0)

    def adjust_position_by_pixel_offset(self, dx, dy, duration=2.0, scale=0.004):
        """
        根据图像偏移调整机体相对偏移飞行

        参数：
            dx: 图像右方向偏移（像素）
            dy: 图像下方向偏移（像素）
            scale: 像素转米的比例
            duration: 持续时间（秒）
        """

        # 计算机体坐标系偏移量
        x_body = -dy * scale  # 图像下为正，机体前为正，符号可能要根据实际调整
        y_body = dx * scale   # 图像右为正，机体右为正

        print(f"[INFO] 机体坐标系偏移: x_body={x_body:.3f}m, y_body={y_body:.3f}m")
        self.fly_toward(x_body, y_body, 0, duration)  # z轴不变，传0表示维持当前高度

    def drop_bottle(self, bottle="front"):
        """
        投掷前/后方水瓶。
        bottle = "front"/"back"
        """
        if bottle == 'front':
            self.fly_toward(0.12, 0, 0, 3)  # 机体坐标系，前进12cm
            self.mav.SetServo(1, -1)
        elif bottle == 'back':
            self.fly_toward(-0.12, 0, 0, 3) # 机体坐标系，后退12cm
            self.mav.SetServo(-1, 1)
        time.sleep(1)

        