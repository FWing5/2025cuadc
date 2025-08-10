import time
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

            print("[SUCCESS] 无人机已连接并准备就绪！")

        except Exception as e:
            print(f"[ERROR] 初始化无人机通信失败: {e}")
            self.mav = None  # 防止self.mav未定义引发后续错误

    def shutdown(self):
        print("[INFO] 上锁无人机（Disarm）...")
        self.mav.SendMavArm(False)
        time.sleep(1)

        print("[INFO] 停止 Offboard 模式...")
        self.mav.stopOffboard()
        time.sleep(1)

        print("[INFO] 停止 MAVLink 通信循环...")
        self.mav.endMavLoop()
        time.sleep(1)

        print("[SUCCESS] 无人机已终止。")

    def get_position(self):
        return self.mav.uavPosNED

    def fly_to(self, x, y, z, duration):
        print(f"飞行至: ({x}, {y}, {z}) ")
        start_time = time.time()
        while time.time() - start_time < duration:
            self.mav.SendPosFRD(x, y, z, 0)
            time.sleep(0.05)  # 控制频率，200ms发送一次

    def hover(self, duration):
        print(f"悬停 {duration} 秒")
        start_time = time.time()
        while time.time() - start_time < duration:
            cur_x, cur_y, cur_z = self.mav.uavPosNED
            self.mav.SendPosFRD(cur_x, cur_y, cur_z, 0)
            time.sleep(0.05)  # 控制频率，200ms发送一次

    def adjust_position_by_pixel_offset(self, dx, dy, current_height, scale=0.004, duration=1.0):
        """
        根据图像中心偏移量进行微调飞行。

        参数：
            dx (int/float): 图像横向偏移（+右，-左）
            dy (int/float): 图像纵向偏移（+下，-上）
            current_height (float): 当前飞行高度（负数）
            scale (float): 像素到米的缩放因子（默认 0.004）
            duration (float): 飞行持续时间（秒）
        """
        if self.mav is None:
            print("[ERROR] 无人机未连接，无法微调")
            return

        # 将图像坐标偏移转换为 NED 坐标调整值
        adjust_north = -dy * scale  # 图像向下是正，NED的north向前
        adjust_east  = dx * scale   # 图像向右是正，NED的east向右

        cur_x, cur_y, _ = self.mav.uavPosNED
        new_x = cur_x + adjust_north
        new_y = cur_y + adjust_east

        print(f"[INFO] 微调坐标：dx={dx}, dy={dy}, 调整后位置=({new_x:.2f}, {new_y:.2f}, {current_height})")
        self.fly_to(new_x, new_y, current_height, duration)

    def move_front_12cm(self):
        """
        投掷前方水瓶。
        """
        x, y, z = self.get_position
        self.fly_to( x + 0.12, y, z, 2)
        self.mav.SetServo
        

    def move_back_12cm(self):
        """
        投掷后方水瓶。
        """
        x, y, z = self.get_position
        self.fly_to( x - 0.12, y, z, 2)