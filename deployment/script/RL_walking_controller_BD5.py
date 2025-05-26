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

class BD5RLController:
    def __init__(
        self,
        onnx_model_path: str,
        DXL_port: str = "/dev/ttyDXL",
        DXL_Baudrate: int = 1000000,
        pitch_bias: float = 0,
        control_freq: float = 50, # 50 Hz
        command_freq: float = 20, # 20 Hz
        exponential_filter: bool = False,
        cutoff_frequency: float = None, # or 40Hz
        history_len: int = 5,
        action_scale: float = 0.3,
        max_motor_speed: float = 4.82,
        pid: float = None,
        vel_range_x: float = [-0.4, 0.4],
        vel_range_y: float = [-0.2, 0.2],
        vel_range_rot: float = [-1.0, 1.0],
        record: bool = False,
        
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
        # Observation memory
        self.obs_history = np.zeros(32 * history_len)

        # Init motor targets
        self.max_motor_speed = max_motor_speed
        self.motor_targets = self._default_angles_leg.copy()
        self.prev_motor_targets = self._default_angles_leg.copy()

        # Time management
        self.control_freq = control_freq
        self._ctrl_dt = 1 / control_freq
        self.smooth_neck = 0.0
        self.tau_neck = 0.4

        # Init joystick
        self.command_freq = command_freq
        self.joystick = Gamepad(command_freq=self.command_freq, 
                                vel_range_x=vel_range_x, 
                                vel_range_y=vel_range_y, 
                                vel_range_rot=vel_range_rot, 
                                head_range=[-0.5236, 0.5236], 
                                deadzone=0.09)
        
        # Init command from joystick
        self.last_command = [0.0, 0.0, 0.0]
        # Init command logic
        self.PAUSED = False

        # Init low pass filter
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(self.control_freq, cutoff_frequency)

        # Exponential filter 
        self.exp_filter = exponential_filter
        self.prev_filter_state = self._default_angles_leg.copy()

        # Init Servo Controller
        self.portHandler = PortHandler(DXL_port)
        self.packetHandler = PacketHandler(2.0)
        # check connection 
        if self.portHandler.openPort():
            print("Successfully opened the port at %s!" % DXL_port)
        else:
            self.portHandler.closePort()
            raise Exception("Failed to open the port at %s!", DXL_port)
        if self.portHandler.setBaudRate(DXL_Baudrate):
            print("Succeeded to change the baudrate to %d bps!" % DXL_Baudrate)
        else:
            self.portHandler.closePort()
            raise Exception("Failed to change the baudrate to %d bps!" % DXL_Baudrate)     

        # init servos class
        self.servo = ServoControllerBD5(portHandler=self.portHandler, packetHandler=self.packetHandler)
        # ping all servos 
        if self.servo.ping():
            print("Servos ready !")
        else:
            self.portHandler.closePort()
            raise Exception("Error in servos ID or state !")   
        
        # set dynamixel PID value 
        if pid is not None:
            self.servo.set_PID(pid=pid)

        # Init IMU
        self.pitch_bias = pitch_bias
        self.imu = IMU(sampling_freq=self.control_freq, user_pitch_bias=self.pitch_bias, calibrate=False)

        # state recorder 
        self.record = record
        if self.record:
            print("start recording data !")
            self.state_data = []

    def start_robot(self):
        print("START BD-5...")
        # enable torque
        self.servo.enable_torque()
        # set default angles
        self.servo.set_position(self._default_angles_full_list)
        time.sleep(2)
        print("BD-5 ready !")
    
    def stop_robot(self):
        print("STOP BD-5...")
        # disable torque
        self.servo.disable_torque()
        time.sleep(2)

    def get_obs(self):
        # get IMU data  
        imu_data = self.imu.get_data()
        # get Dynamixel data 
        dxl_qpos, success_pos = self.servo.get_position(full=False) # Only recover the state of the legs
        if not success_pos or len(dxl_qpos) == 0:
            return None
        current_qpos = np.array(dxl_qpos) # Only recover the state of the legs
        # get joint angles delta and velocities
        joint_angles = current_qpos - self._default_angles_leg
        # concatenate all
        obs = np.hstack([
            imu_data["gyro"],
            imu_data["accelerometer"],
            imu_data["gravity"],
            self.last_command,
            joint_angles,
            self._last_action,
        ])
        # update history memory
        state_size = obs.shape[0]
        # fill the buffer
        self.obs_history = np.roll(self.obs_history, state_size)
        self.obs_history[:state_size] = obs
        # record state if needed
        if self.record:
            if obs is not None:
                self.state_data.append(self.obs_history)
        return self.obs_history.astype(np.float32)
    
    def run(self):
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
                controlled_neck = [self._default_angles_head_list[0], self._default_angles_head_list[1] + self.smooth_neck]
                # Kill-switch exit program
                if X_pressed == True:
                    self.stop_robot()
                    self.portHandler.closePort()
                    print("Port closed !")
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
                onnx_pred = self.policy.infer(onnx_input)
                # update action memory
                self._last_action = onnx_pred.copy()
                # update motor targets
                self.motor_targets = onnx_pred * self._action_scale + self._default_angles_leg
                # filter the motor output 
                if self.exp_filter:
                    filter_state = 0.8 * self.prev_filter_state + 0.2 * self.motor_targets
                    self.prev_filter_state = filter_state.copy()
                    self.motor_targets = filter_state
                # get action filtered 
                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (time.time() - start_t > 1):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets
                # clip motor speed 
                if self.max_motor_speed is not None:
                    self.motor_targets = np.clip(self.motor_targets, 
                                                self.prev_motor_targets - self.max_motor_speed * (self._ctrl_dt),
                                                self.prev_motor_targets + self.max_motor_speed * (self._ctrl_dt)
                                                )
                # update previous motor targets
                self.prev_motor_targets = self.motor_targets.copy()
                # send motor target to servos 
                target_position = self.motor_targets.tolist() + controlled_neck 
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
            print("KeyboardInterrupt detected !")
            self.stop_robot()
            self.portHandler.closePort()
            print("Port closed !")
            pass
        if self.record:
            self.state_data = np.array(self.state_data)
            np.save("/home/robot/BD-5/bd5_state.npy", self.state_data)
            print("state recorded !")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("--action_scale", type=float, default=0.3)
    parser.add_argument("--control_freq", type=int, default=50)
    parser.add_argument("--command_freq", type=int, default=20)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument("--max_motor_speed", type=float, default=4.50)
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--pid", type=float, default=None)
    parser.add_argument("--vel_range_x", type=float, nargs=2, default=[-0.8, 0.8])
    parser.add_argument("--vel_range_y", type=float, nargs=2, default=[-0.4, 0.4])
    parser.add_argument("--vel_range_rot", type=float, nargs=2, default=[-0.8, 0.8])
    parser.add_argument("--DXL_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--DXL_Baudrate", type=int, default=1000000)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--exponential_filter", type=bool, default=False)
    parser.add_argument("--record", type=bool, default=False)

    args = parser.parse_args()

    BD5_ctrl = BD5RLController(
        onnx_model_path=args.onnx_model_path,
        DXL_port=args.DXL_port,
        DXL_Baudrate=args.DXL_Baudrate,
        pitch_bias=args.pitch_bias,
        control_freq=args.control_freq,
        command_freq=args.command_freq,
        exponential_filter=args.exponential_filter,
        cutoff_frequency=args.cutoff_frequency,
        history_len=args.history_len,
        action_scale=args.action_scale,
        max_motor_speed=args.max_motor_speed,
        vel_range_x=args.vel_range_x,
        vel_range_y=args.vel_range_y,
        vel_range_rot=args.vel_range_rot,
        record=args.record,
    )

    print("BD-5 RL Controller initialized !")
    BD5_ctrl.run()
