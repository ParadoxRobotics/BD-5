import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
IMU = BNO08X_I2C(i2c)

# Get quaternion mode 
IMU.enable_feature(BNO_REPORT_ROTATION_VECTOR)

while True:
    try:
        # read
        quat_i, quat_j, quat_k, quat_real = IMU.quaternion
        quaternion = np.array([quat_i, quat_j, quat_k, quat_real])
        # quaternion to euler 
        euler = R.from_quat(quat=quaternion, scalar_first=False).as_euler("xyz")
        print(quaternion, euler)
    except Exception as e:
        print(e)
        continue

    time.sleep(1 / 30)