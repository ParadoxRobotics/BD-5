import os
import time
import numpy as np

from dynamixel_sdk import *

from CTRL.Servo_Controller_BD5 import ServoControllerBD5
from CTRL.IMU import IMU
from CTRL.ONNX_infer import OnnxInfer
from CTRL.Gamepad import Gamepad

class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency=40.0):
        self.last_action = 0
        self.current_action = 0
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
        )

    def push(self, action):
        self.current_action = action

    def get_filtered_action(self):
        self.last_action = (
            self.alpha * self.last_action + (1 - self.alpha) * self.current_action
        )
        return self.last_action

class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        DXL_port: str = "/dev/ttyDXL",
        DXL_Baudrate: int = 1000000,
        pitch_bias: float = 0,
        control_freq: float = 50, # 50 Hz
        command_freq: float = 20, # 20 Hz
        action_scale: float = 0.3,
        gait_freq: float = 1.0,
        max_motor_speed: float = 4.82,
        vel_range_x: float = [-0.6, 0.6],
        vel_range_y: float = [-0.6, 0.6],
        vel_range_rot: float = [-1.0, 1.0],
        cutoff_frequency=40, # or 40Hz
    ):
        # Init Model 
        self.model_path = onnx_model_path
        self.policy = OnnxInfer(self.model_path)
        print("Policy Model Loaded !")

        # Init action scale
        self._action_scale = action_scale

        # Init pose 
        self._default_angles_leg_list = [0.0, 
                                        0.0, 
                                        0.82498, 
                                        1.64996,
                                        0.82498,
                                        0.0,
                                        0.0,
                                        0.82498,
                                        1.64996,
                                        0.82498]
        self._default_angles_leg = np.array(self._default_angles_leg_list)
        self._default_angles_head_list = [0.5306, -0.5306]
        self._default_angles_full_list = self._default_angles_leg_list + self._default_angles_head_list
        
        # Action memory
        self._last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)
        self._last_last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)
        self._last_last_last_action = np.zeros_like(self._default_angles_leg, dtype=np.float32)

        # Init motor targets
        self.max_motor_speed = max_motor_speed
        self.motor_targets = self._default_angles_leg.copy()
        self.prev_motor_targets = self._default_angles_leg.copy()

        # Time management
        self._ctrl_dt = 1 / control_freq
        self.smooth_neck = 0.0
        self.tau_neck = 0.4

        # Phase init -> in real case self._ctrl_dt = self._n_substeps * self._sim_dt
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = gait_freq
        self._phase_dt = 2 * np.pi * self._gait_freq * self._ctrl_dt 

        # Init joystick
        self.command_freq = command_freq
        self.joystick = Gamepad(command_freq=self.command_freq, 
                                vel_range_x=vel_range_x, 
                                vel_range_y=vel_range_y, 
                                vel_range_rot=vel_range_rot, 
                                head_range=[-0.5236, 0.5236], 
                                deadzone=0.05)
        
        # Init command from joystick
        self.last_command = [0.0, 0.0, 0.0]
        self.last_head_tilt = 0.0
        # Init command logic
        self.PAUSED = False

        # Init low pass filter
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(self.control_freq, cutoff_frequency)

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
        print("START BD-5...")
        # enable torque
        self.servo.enable_torque()
        # set default angles
        self.servo.set_position(self._default_angles_full_list)
    
    def stop_robot(self):
        print("STOP BD-5...")
        # disable torque
        self.servo.disable_torque()
        time.sleep(2)

    def get_obs(self):
        # get IMU data  
        imu_data = self.imu.get_data()
        # get Dynamixel data 
        dxl_qpos, success = self.servo.get_position()
        dxl_qvel, success = self.servo.get_velocity()
        if len(dxl_qpos) == 0 or len(dxl_qvel) == 0:
            return None
        current_qpos = np.array(dxl_qpos[:2])
        current_qvel = np.array(dxl_qvel[:2])
        # get joint angles delta and velocities
        joint_angles = current_qpos - self._default_angles_leg
        joint_velocities = current_qvel
        # adjust phase
        ph = self._phase if np.linalg.norm(self.last_command) >= 0.01 else np.ones(2) * np.pi
        phase = np.concatenate([np.cos(ph), np.sin(ph)])
        # concatenate all
        obs = np.hstack([
            imu_data["gyro"],
            imu_data["accelorometer"],
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
        # TODO : Wait for joystick connection
        # Wait to start the BD-5
        while True:
            self.last_command, head_tilt, S_pressed, T_pressed, C_pressed, X_pressed = self.joystick.get_last_command()
            if C_pressed == True:
                break
        self.start_robot()
        # Start main loop
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                t = time.time()
                # get command from joystick
                self.last_command, head_tilt, S_pressed, T_pressed, C_pressed, X_pressed = self.joystick.get_last_command()
                # get head tilt command 
                self.smooth_neck = self.tau_neck * head_tilt + (1 - self.tau_neck) * self.smooth_neck
                controlled_neck = [self._default_angles_head[0], self._default_angles_head[1] + self.smooth_neck]
                # Kill-switch exit program
                if X_pressed == True:
                    self.ENABLE = False
                    self.stop_robot()
                    break
                # Pause inference/action process 
                if T_pressed == True:
                    self.PAUSED = not self.PAUSED
                    if self.PAUSED:
                        print("PAUSE")
                    else:
                        print("UNPAUSE")
                if self.PAUSED:
                    time.sleep(0.01)
                    continue
                # get observation 
                obs = self.get_obs()
                if obs is None:
                    print("No observation")
                    continue
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
                # get action filtered 
                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (time.time() - start_t > 1):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets
                # update previous motor targets
                self.prev_motor_targets = self.motor_targets.copy()
                # send motor target to servos 
                target_position = list(self.motor_targets) + controlled_neck 
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
        except KeyboardInterrupt:
            pass
