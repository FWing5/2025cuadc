import time
from drone_controller import *

drone = DroneController()

drone.fly_to(0, 0, -4, 10)

drone.fly_to(2, 0, -4, 10)

drone.fly_to(2, 2, -4, 10)

drone.fly_to(0, 0, -4, 10)

drone.land()

time.sleep(10)
drone.shutdown()