import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from etils import epath
from common.onnx_infer import OnnxInfer

import base

class BD5_Infer:
    def __init__(
        self, model_path: str, onnx_model_path: str, gait_freq: float, dt: float):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.action_scale = 0.5

        NUM_DOFS = 10

        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-1.0, 1.0]
        self.COMMANDS_RANGE_Y = [-1.0, 1.0]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.last_action = np.zeros(NUM_DOFS)
        self.last_last_action = np.zeros(NUM_DOFS)
        self.last_last_last_action = np.zeros(NUM_DOFS)
        self.commands = [0.0, 0.0, 0.0] # Vx, Vy, Th
        self.decimation = 10

        # Init gait phase and frequency
        phase_dt = 2 * np.pi * dt * gait_freq
        phase_init = np.array([0, np.pi])
        phase_tp1 = phase_init + phase_dt
        phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        cos = np.cos(phase)
        sin = np.sin(phase)
        self.phase = np.concatenate([cos, sin])

        # Init observation memory
        self.saved_obs = []

        # Recover init pose
        self.init_q = np.array(self._mj_model.keyframe("init_pose").qpos)
        self.default_pose = self._mj_model.keyframe("init_pose").ctrl

        # Init pose and actuator 
        self.data.qpos[:] = self.model.keyframe("init_pose").qpos
        self.data.ctrl[:] = self.default_actuator

        # Gyro description
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.gyro_addr = self.model.sensor_adr[self.gyro_id]
        self.gyro_dimensions = 3

        # Accelerometer description
        self.accelerometer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
        self.accelerometer_addr = self.model.sensor_adr[self.accelerometer_id]
        self.accelerometer_dimensions = 3
        
        # Imu description
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

    def get_sensor(self, data, name, dimensions):
        i = self.model.sensor_name2id(name)
        return data.sensordata[i : i + dimensions]

    def get_gyro(self, data):
        return data.sensordata[self.gyro_addr : self.gyro_addr + self.gyro_dimensions]

    def get_accelerometer(self, data):
        return data.sensordata[self.accelerometer_addr : self.accelerometer_addr + self.accelerometer_dimensions]

    def get_gravity(self, data):
        return data.site_xmat[self.imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])

    def get_obs(
        self,
        data,
        last_action,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        # Recover current state
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        gravity = self.get_gravity(data)
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        # keycode 
        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]
        # init command 
        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel
        print(self.commands)

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()
                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:
                        # get observation
                        obs = self.get_obs(
                            self.data,
                            self.last_action,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        action = self.default_actuator + action * self.action_scale

                        self.data.ctrl = action.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/master/Bureau/BD-5/RL/xmls/scene_flat.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = BD5_Infer(args.model_path, args.onnx_model_path, gait_freq=1.25, dt= 0.002)
    mjinfer.run()