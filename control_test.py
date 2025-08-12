import time
from drone_controller import *

drone = DroneController()

drone.fly_to(0, 0, -4, 2)

drone.fly_to(3, 0, -4, 3)

drone.hover(5)

drone.fly_to(3, 3, -4, 3)

drone.fly_to(0, 0, -4, 2)

drone.land()

time.sleep(10)
drone.shutdown()