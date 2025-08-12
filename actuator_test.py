import time
from drone_controller import *

drone = DroneController()

time.sleep(5)
drone.drop_bottle("front")
time.sleep(5)
drone.drop_bottle("back")

time.sleep(10)
drone.shutdown()