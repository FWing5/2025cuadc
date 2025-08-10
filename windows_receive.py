import cv2
import requests
import numpy as np
import socket
import time

IP_BASE = "192.168.4."
PORT = 8000
TIMEOUT = 1  # 秒

def check_port(ip, port=PORT, timeout=TIMEOUT):
    """检测ip:port是否开放"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False

def find_working_ip():
    """扫描192.168.4.2~4，返回第一个能连上的IP"""
    for i in range(2, 5):
        ip = IP_BASE + str(i)
        print(f"检测 {ip}:{PORT} ...")
        if check_port(ip, PORT):
            print(f"找到可用IP: {ip}")
            return ip
    print("未找到可用IP")
    return None

def get_latest_frame(ip):
    url = f"http://{ip}:{PORT}/latest_frame"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"获取图像失败: {e}")
        return None

def main():
    ip = find_working_ip()
    if ip is None:
        print("程序结束，未找到有效服务器。")
        return

    print("开始显示远程图像，按 q 退出")
    while True:
        frame = get_latest_frame(ip)
        if frame is not None:
            cv2.imshow("远程最新画面", frame)
        else:
            cv2.imshow("远程最新画面", np.zeros((480, 640, 3), dtype=np.uint8))

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
