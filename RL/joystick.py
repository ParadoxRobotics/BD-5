# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Joystick task for the BD-5
# See Booster T1 in the locomotion folder for more details
# And OpenDuck for value range

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait, mjx_env
from mujoco_playground._src.collision import geoms_colliding
import base as BD5_base
import constants as consts

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=500,
      action_repeat=1,
      action_scale=1.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          action_min_delay=0,  # env steps
          action_max_delay=3,  # env steps
          imu_min_delay=0,  # env steps
          imu_max_delay=3,  # env steps
          scales=config_dict.create(
              hip_pos=0.03,
              knee_pos=0.05,
              ankle_pos=0.08,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=-1.0e-3,
              action_rate=-0.75,
              energy=0.0,
              dof_acc=0.0,
              dof_vel=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              stand_still=0.0,
              alive=0.25,
              termination=0.0,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.1,
              dof_pos_limits=-1.0,
              pose=-1.0,
              feet_distance=-1.0,
              collision=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.1,
          base_height_target=0.5,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-1.0, 1.0],
      ang_vel_yaw=[-1.0, 1.0],
  )

class Joystick(BD5_base.BD5Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )

    # Initialize
    self._post_init()

  def _post_init(self) -> None:
    # Init default pose
    self._init_q = jp.array(self._mj_model.keyframe("init_pose").qpos)
    self._default_pose = jp.array(self.get_all_joints_qpos(self._mj_model.keyframe("init_pose")))
    self._default_actuator = self._mj_model.keyframe("init_pose").ctrl

    # Get the range of the joints
    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    hip_indices = []
    hip_indices.append(self._mj_model.joint("left_hip_roll").qposadr - 7)
    hip_indices.append(self._mj_model.joint("right_hip_roll").qposadr - 7)
    hip_indices.append(self._mj_model.joint("left_hip_yaw").qposadr - 7)
    hip_indices.append(self._mj_model.joint("right_hip_roll").qposadr - 7)
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    knee_indices.append(self._mj_model.joint("left_knee").qposadr - 7)
    knee_indices.append(self._mj_model.joint("right_knee").qposadr - 7)
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array(
        [
            1.0, # left_hip_yaw
            1.0, # left_hip_roll
            0.01, # left_hip_pitch
            0.01, # left_knee
            1.0,  # left_ankle
            1.0, # right_hip_yaw
            1.0, # right_hip_roll
            0.01, # right_hip_pitch
            0.01, # right_knee
            1.0,  # right_ankle
        ]
    )
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._left_feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS])
    self._right_feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS])
    self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._left_foot_box_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_box_geom_id = self._mj_model.geom("right_foot").id