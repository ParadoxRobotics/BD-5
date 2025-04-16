import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GRAVITY
)

import time
from queue import Queue
from threading import Thread
import numpy as np
from scipy.spatial.transform import Rotation as R


class IMU:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False):
        self.sampling_freq = sampling_freq
        self.user_pitch_bias = user_pitch_bias
        self.nominal_pitch_bias = 0
        self.calibrate = calibrate

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.imu = BNO08X_I2C(i2c)

        # Enable Gyroscope, Accelerometer and Rotation Vector
        self.imu.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.imu.enable_feature(BNO_REPORT_GYROSCOPE)
        self.imu.enable_feature(BNO_REPORT_GRAVITY)

        self.pitch_bias = self.nominal_pitch_bias + self.user_pitch_bias

        # IMU calibration 
        if self.calibrate:
            # start calibration
            self.imu.begin_calibration()
            calibration_good_at = None
            while True:
                time.sleep(0.1)
                calibration_status = self.imu.calibration_status
                print("Calibration status: ", calibration_status)
                if not calibration_good_at and calibration_status >= 2:
                    calibration_good_at = time.monotonic()
                    if calibration_good_at and (time.monotonic() - calibration_good_at > 5.0):
                        input_str = input("\n\nEnter S to save or anything else to continue: ")
                        if input_str.strip().lower() == "s":
                            self.imu.save_calibration_data()
                            break
                        calibration_good_at = None
            print("Calibration complete! Exiting...")
            exit()
        
        self.last_imu_data = {
            "gyro": [0, 0, 0],
            "accelerometer": [0, 0, 0],
            "gravity": [0, 0, 0],
        }
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                # get data 
                gyro = np.array(self.imu.gyro).copy()
                accelerometer = np.array(self.imu.acceleration).copy()
                gravity = np.array(self.imu.gravity).copy()
                gravity = gravity / np.linalg.norm(gravity) 
                gravity[2] = -gravity[2]
                 
            except Exception as e:
                print("[IMU]:", e)
                continue

            if gyro is None or accelerometer is None or gravity is None:
                continue

            if gyro.any() is None or accelerometer.any() is None or gravity.any() is None:
                continue

            data = {
                "gyro": gyro,
                "accelerometer": accelerometer,
                "gravity": gravity,
            }

            self.imu_queue.put(data)
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self, euler=False, mat=False):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data


if __name__ == "__main__":
    imu = IMU(50, calibrate=False)
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("accelerometer", np.around(data["accelerometer"], 3))
        print("gravity", np.around(data["gravity"], 3))
        print("---")
        time.sleep(1 / 25)