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

# Base class for the BD-5 droid (joint, sensor,...)
# See Booster T1 in the locomotion folder for more details
# And OpenDuck for joint access 

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
import constants


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls" / "assets")
    path = constants.ROOT_PATH
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class BD5Env(mjx_env.MjxEnv):
    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        print(f"xml: {xml_path}")
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        
    # Sensor readings.
    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Return the gravity vector in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, constants.GRAVITY_SENSOR)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GYRO_SENSOR
        )

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        """Return the position of the feet in the world frame."""
        return jp.vstack(
            [mjx_env.get_sensor_data(self.mj_model, data, sensor_name) for sensor_name in constants.FEET_POS_SENSOR]
        )


    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
