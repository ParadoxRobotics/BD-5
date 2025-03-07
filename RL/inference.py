import mujoco
import pickle
import numpy as np

import mujoco.viewer
import time
import argparse

from common.onnx_infer import OnnxInfer

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("-k", action="store_true", default=False)
args = parser.parse_args()

NUM_JOINTS = 10
DT = 0.002

if args.k:
    import pygame
    pygame.init()
    # open a blank pygame window
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press arrow keys to move robot")

# Params
linearVelocityScale = 1.0
angularVelocityScale = 1.0
dof_pos_scale = 1.0
dof_vel_scale = 1.0
action_scale = 0.5

# Init position
init_pos = np.array(
    [
      0.0,
      0.0,
      0.82498,
      1.64996,
      0.82498,
      0.0,
      0.0,
      0.82498,
      1.64996,
      0.82498,
    ]
)

# Load environment
model = mujoco.MjModel.from_xml_path(
    "/home/master/Bureau/BD-5/RL/xmls/scene_flat.xml"
)

# Init simulation
model.opt.timestep = DT
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

# Load policy
policy = OnnxInfer(args.onnx_model_path, awd=True)

COMMANDS_RANGE_X = [-0.1, 0.15]
COMMANDS_RANGE_Y = [-0.2, 0.2]
COMMANDS_RANGE_THETA = [-0.5, 0.5] # [-1.0, 1.0]

# Init state
last_action = np.zeros(NUM_JOINTS)
last_last_action = np.zeros(NUM_JOINTS)
last_last_last_action = np.zeros(NUM_JOINTS)
commands = [0.0, 0.0, 0.0]
decimation = 10
data.qpos[3 : 3 + 4] = [1.0, 0.0, 0.0, 0.0]

# Init gait phase and frequency
gait_freq = 1.25
phase_dt = 2 * np.pi * DT * gait_freq
phase_init = np.array([0, np.pi])
phase_tp1 = phase_init + phase_dt
phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
cos = np.cos(phase)
sin = np.sin(phase)
phase = np.concatenate([cos, sin])

data.qpos[7:] = init_pos
data.ctrl[:] = init_pos

# IMU fusion
imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
# Gyroscope
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
gyro_addr = model.sensor_adr[gyro_id]
gyro_dim = 3
# Accelerometer
accelerometer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
accelerometer_addr = model.sensor_adr[accelerometer_id]
accelerometer_dim = 3

def get_sensor(model, data, name, dimensions):
    i = model.sensor_name2id(name)
    return data.sensordata[i : i + dimensions]

def get_gravity(data):
    return data.site_xmat[imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])

def get_gyro(data):
    return data.sensordata[gyro_addr : gyro_addr + gyro_dim]

def get_accelerometer(data):
    return data.sensordata[accelerometer_addr : accelerometer_addr+ accelerometer_dim]

def get_obs(data, last_action, last_last_action, last_last_last_action, command, phase):
    # get gyro, accelerometer, gravity, joint pos and vel
    gyro = get_gyro(data)
    accelerometer = get_accelerometer(data)
    gravity = get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    # observation vector 
    obs = np.concatenate(
        [
            gyro,
            accelerometer,
            gravity,
            command,
            joint_angles - init_pos,
            joint_vel * dof_vel_scale,
            last_action,
            last_last_action,
            last_last_last_action,
            phase,
        ]
    )
    return obs

def key_callback(keycode):
    pass

def handle_keyboard():
    global commands
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keys[pygame.K_z]:
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keys[pygame.K_s]:
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keys[pygame.K_q]:
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keys[pygame.K_d]:
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keys[pygame.K_a]:
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keys[pygame.K_e]:
        ang_vel = COMMANDS_RANGE_THETA[0]

    commands[0] = lin_vel_x
    commands[1] = lin_vel_y
    commands[2] = ang_vel
    print(commands)
    pygame.event.pump()  # process event queue


saved_obs = []
try:
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer:
        counter = 0
        while True:

            step_start = time.time()

            mujoco.mj_step(model, data)

            counter += 1

            if counter % decimation == 0:
                # update observation
                obs = get_obs(
                    data,
                    last_action,
                    commands,
                    phase
                )
                saved_obs.append(obs)
                action = policy.infer(obs)
                # update memory
                last_last_last_action = last_last_action.copy()
                last_last_action = last_action.copy()
                last_action = action.copy()
                # action = np.zeros(10)
                action = init_pos + action * action_scale
                data.ctrl = action.copy()

            viewer.sync()

            if args.k:
                handle_keyboard()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))