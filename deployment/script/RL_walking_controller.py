import os
import time
import numpy as np

from dynamixel_sdk import *

from CTRL.Servo_Controller_BD5 import ServoControllerBD5
from CTRL.IMU import IMU
from CTRL.ONNX_infer import OnnxInfer
from CTRL.Gamepad import Gamepad

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
        # TODO : add default head position 
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
        self.joystick = Gamepad(command_freq=control_freq, vel_range_x=vel_range_x, vel_range_y=vel_range_y, vel_range_rot=vel_range_rot, deadzone=0.04)
        self.last_command = [0.0, 0.0, 0.0]
        self.ENABLE = False

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

    def start_robot(self):
        # enable torque
        self.servo.enable_torque()
        time.sleep(2)
        # set default angles
        self.servo.set_position(self._default_angles_full)
        time.sleep(2)
    
    def stop_robot(self):
        # disable torque
        self.servo.disable_torque()
        time.sleep(2)

    def get_obs(self):
        # get IMU data  
        imu_data = self.imu.get_data()
        # get Dynamixel data
        current_qpos, success = self.servo.get_position()
        current_qvel, success = self.servo.get_velocity()
        current_qpos = np.array(current_qpos)
        current_qvel = np.array(current_qvel)
        # get joint angles delta and velocities
        joint_angles = current_qpos - self._default_angles_leg
        joint_velocities = current_qvel
        # adjust phase
        ph = self._phase if np.linalg.norm(self.last_command) >= 0.01 else np.ones(2) * np.pi
        phase = np.concatenate([np.cos(ph), np.sin(ph)])
        # concatenate all
        obs = np.hstack([
            imu_data["gyro"],
            imu_data["acceleration"],
            # imu_data["orientation"],
            self.last_command,
            joint_angles,
            joint_velocities,
            self._last_action,
            self._last_last_action,
            self._last_last_last_action,
            phase,
        ])
        return obs.astype(np.float32)
    
    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                t = time.time()
                # get command from joystick
                self.last_command, head_tilt, Akey, Xkey, Bkey, Ykey = self.joystick.get_last_command()
                # Activate the robot
                if Akey == True:
                    self.ENABLE = True
                    self.start_robot()
                # Kill-switch
                if Xkey == True:
                    self.ENABLE = False
                    self.stop_robot()
                    break
                # Pause inference/action process 
                if Bkey == True:
                    time.sleep(0.1)
                    continue
                
                if self.ENABLE == True:
                    # get observation 
                    obs = self.get_obs()
                    onnx_input = {"obs": obs.reshape(1, -1)}
                    # Policy inference 
                    onnx_pred = self.policy(onnx_input)
                    # update action memory
                    self._last_last_last_action = self._last_last_action.copy()
                    self._last_last_action = self._last_action.copy()
                    self._last_action = onnx_pred.copy()
                    # update motor targets -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
                    self.motor_targets = onnx_pred * self._action_scale + self._default_angles_leg
                    self.motor_targets = np.clip(self.motor_targets, 
                                                self.prev_motor_targets - self.max_motor_speed * (self._ctrl_dt),
                                                self.prev_motor_targets + self.max_motor_speed * (self._ctrl_dt)
                                                )
                    self.prev_motor_targets = self.motor_targets.copy()
                    # send motor target to servos # TODO : add head control
                    target_position = list(self.motor_targets) + self._default_angles_head 
                    self.servo.set_position(value=target_position)
                    # time control 
                    i+=1
                    took = time.time() - t
                    if (1 / self.control_freq - took) < 0:
                        print(
                            "Policy control budget exceeded by",
                            np.around(took - 1 / self.control_freq, 3),
                        )
                    time.sleep(max(0, 1 / self.control_freq - took))
                else:
                    continue

        except KeyboardInterrupt:
            pass
