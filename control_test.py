import time
import PX4MavCtrlV4 as PX4MavCtrl

# 初始化PX4控制器
mav = PX4MavCtrl.PX4MavCtrler(1, Com = '/dev/ttyUSB0')
mav.InitMavLoop(2)
time.sleep(1)

print('进入offboard并解锁')
mav.initOffboard()
time.sleep(1)

# 飞机起飞
print('发送起飞命令')
mav.sendMavTakeOff(0,0,4)
time.sleep(10)

mav.SendPosFRD(2,0,4)
time.sleep(10)

mav.sendMavLand(0, 0, 0)
time.sleep(10)

mav.endOffboard()
time.sleep(1)
mav.endMavLoop()