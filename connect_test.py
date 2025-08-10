import time
import PX4MavCtrlV4 as PX4MavCtrl

# mav = PX4MavCtrl.PX4MavCtrler(1, Com = '/dev/ttyUSB0')
mav = PX4MavCtrl.PX4MavCtrler(1, Com = 'COM3')

mav.InitMavLoop(2)
print(mav.the_connection)
print("arm")
mav.SendMavArm(True)
time.sleep(5)
print("disarm")
mav.SendMavArm(False)
mav.stopRun()