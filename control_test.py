import time
from drone_controller import *

drone = DroneController()

try:
    drone.fly_to(0, 0, -2, 5)
    drone.fly_to(3, 0, 4, 5)

    drone.land()
except Exception as e:
    print(f"出现异常: {e}，执行紧急降落！")
    try:
        drone.land()
    except Exception as land_e:
        print(f"降落过程中也发生异常: {land_e}")
    # 可以选择是否继续抛出异常，或直接退出
    # raise
finally:
    time.sleep(10)
    try:
        drone.shutdown()
    except Exception as shutdown_e:
        print(f"关闭飞控时异常: {shutdown_e}")
