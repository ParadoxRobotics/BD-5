import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE

# Integration dynamics
dt = 0.01

# Pendulum dynamics 
m = 1.907 # mass of the body in kg
l = 0.16 # distance to ground in m
g = 9.81 # grav acc m/s**2

# Coupling constant
kc = 9.4 # must respect |wc - wr| < Kc + Kr 
kp = kc/2
# natural frenquency of the pendulum 
wc = np.sqrt(g/l) 

# Phase difference
alpha_c = np.array([[-1/2*np.pi], [1/2*np.pi]]) # {alphac1, alphac2}
alpha_p = np.array([[0.0], [np.pi]]) # {alphap1, alphap2}

# Amplitude 
Ar = 7.5 # amplitude for the hip roll
Ap = 3.5 # amplitude for the hip-knee-ankle pitch
As_h = 12.0 # amplitude for the hip swing
As_a = 8.0 # amplitude for the ankle swing

# Rest angle 
theta_r = 0
theta_p = 0

# Init
imu_gyro_ang = 0
imu_gyro_vel = 0
phi_c = 0
phi_p = 0
SUP = False # Single support SUP = True, Double support SUP = False 

# connect IMU to I2C 
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
IMU = BNO08X_I2C(i2c)
# Get quaternion and velocity mode 
IMU.enable_feature(BNO_REPORT_ROTATION_VECTOR)
IMU.enable_feature(BNO_REPORT_GYROSCOPE)

for i in range(1000):
    # Measure support and IMU data
    try:
        # read
        quat_i, quat_j, quat_k, quat_real = IMU.quaternion
        gyro_x, gyro_y, gyro_z = IMU.gyro
        quaternion = np.array([quat_i, quat_j, quat_k, quat_real])
        # quaternion to euler 
        euler = R.from_quat(quat=quaternion, scalar_first=False).as_euler("xyz")
        # measurement on the y axis 
        imu_gyro_ang = euler[1]
        imu_gyro_vel = gyro_y
        # compute lateral COM projection given IMU measurement phir = -arctan(ycom_dot_proj/ycom_proj)
        phase_rob = -np.arctan2((l*np.cos(imu_gyro_ang)*imu_gyro_vel), (l*np.sin(imu_gyro_ang))) 
        # compute coordination oscillator 
        phi_c_dot = wc + kc * np.sin((phase_rob - alpha_c) - phi_c)
        # compute step oscillator 
        if SUP:
            phi_p_dot = wc + kp * np.sin((phase_rob - alpha_p) - phi_p)
        else:
            phi_p_dot = 0
        # phase integration 
        phi_c += phi_c_dot * dt
        phi_p += phi_p_dot * dt
        # side-to-side controller
        theta_hip_roll_left = Ar * np.sin(phi_c[0]) + theta_r
        theta_hip_roll_right = theta_hip_roll_left
        # Foot clearance left 
        theta_hip_pitch_left = Ap * np.sin(phi_c[1]) + theta_p
        theta_knee_pitch_left = -2*Ap * np.sin(phi_c[1]) - 2*theta_p
        theta_ankle_pitch_left = -Ap * np.sin(phi_c[1]) - theta_p
        # Foot clearance right
        theta_hip_pitch_right = Ap * np.sin(phi_c[0]) + theta_p
        theta_knee_pitch_right = -2*Ap * np.sin(phi_c[0]) - 2*theta_p
        theta_ankle_pitch_right = -Ap * np.sin(phi_c[0]) - theta_p
        # Foot swing left
        theta_hip_swing_left = As_h * np.sin(phi_p[1])
        theta_ankle_swing_left = -As_a * np.sin(phi_p[1])
        # Foot swing right 
        theta_hip_swing_right = As_h * np.sin(phi_p[0])
        theta_ankle_swing_right = -As_a * np.sin(phi_p[0])
        # total Foot swing left
        theta_hip_pitch_left += theta_hip_swing_left
        theta_ankle_pitch_left += theta_ankle_swing_left
        # total foot swing right
        theta_hip_pitch_right += theta_hip_swing_right
        theta_ankle_pitch_right += theta_ankle_swing_right        
    except Exception as e:
        print(e)
        continue

