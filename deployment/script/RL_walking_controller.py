import os
import time
import numpy as np

from dynamixel_sdk import *

from CTRL.Servo_Controller_BD5 import ServoControllerBD5
from CTRL.IMU import IMU
from CTRL.ONNX_infer import OnnxInfer

joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "neck_pitch",
    "head_pitch"
]

class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        DXL_port: str = "/dev/ttyDXL",
        DXL_Baudrate: int = 1000000,
        pitch_bias: float = 0,
        control_freq: float = 50,
        action_scale: float =0.3,
        gait_freq: float = 1.25,
        max_motor_speed: float = 4.82,
        vel_range_x: float = [-1.0, 1.0],
        vel_range_y: float = [-1.0, 1.0],
        vel_range_rot: float = [-1.0, 1.0],
    ):
        # Init Model 
        self.model_path = onnx_model_path
        self.policy = OnnxInfer(self.model_path)

        # Init action scale and memory
        self._action_scale = action_scale
        self._default_angles_leg = [0.0, 
                                    0.0, 
                                    0.82498, 
                                    1.64996,
                                    0.82498,
                                    0.0,
                                    0.0,
                                    0.82498,
                                    1.64996,
                                    0.82498]
        self._default_angles_head = [0.0, 0.0]
        self._default_angles_full = self._default_angles_leg + self._default_angles_head

        self._last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)
        self._last_last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)
        self._last_last_last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)

        # Init motor targets
        self.max_motor_speed = max_motor_speed
        self.motor_targets = self._default_angles_leg
        self.prev_motor_targets = self._default_angles_leg

        # Time management
        self._ctrl_dt = 1 / control_freq

        # Phase init -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = gait_freq
        self._phase_dt = 2 * np.pi * self._gait_freq * self._ctrl_dt 

        # Init joystick
        self.last_command = [0.0, 0.0, 0.0]

        # Init Servo Controller
        portHandler = PortHandler(DXL_port)
        packetHandler = PacketHandler(2.0)
        # check connection 
        if portHandler.openPort():
            print("Successfully opened the port at %s!" % DXL_port)
        else:
            portHandler.closePort()
            raise Exception("Failed to open the port at %s!", DXL_port)
        if portHandler.setBaudRate(DXL_Baudrate):
            print("Succeeded to change the baudrate to %d bps!" % DXL_Baudrate)
        else:
            portHandler.closePort()
            raise Exception("Failed to change the baudrate to %d bps!" % DXL_Baudrate)
        # init servos class
        self.servo = ServoControllerBD5(portHandler=portHandler, packetHandler=packetHandler)

        # Init IMU
        self.pitch_bias = pitch_bias
        self.imu = IMU(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
        )

    def init_robot(self):
        # enable torque
        self.servo.enable_torque()
        time.sleep(2)
        # set default angles
        self.servo.set_position(self._default_angles_full)
        time.sleep(2)

    def get_obs(self):
        # get IMU data  
        data = self.imu.get_data()
        # get Dynamixel data
        current_qpos, success = self.servo.get_position()
        current_qvel, success = self.servo.get_velocity()
        # get command
        command = None
        # get joint angles delta and velocities
        joint_angles = current_qpos - self._default_angles_leg
        joint_velocities = current_qvel
        # TODO : WTF is that !!!!!!!
        joint_angles[:2] *= 0.0
        joint_velocities[:2] *= 0.0
        # adjust phase
        ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
        phase = np.concatenate([np.cos(ph), np.sin(ph)])
        # concatenate all
        obs = np.hstack([
            data["gyro"],
            data["accelero"],
            #data["orientation"],
            command,
            joint_angles,
            joint_velocities,
            self._last_action,
            self._last_last_action,
            self._last_last_last_action,
            phase,
        ])
        return obs.astype(np.float32)
    
    def control_loop(self):
        i = 0
