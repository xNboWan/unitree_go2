# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

   
def reward_tracking_contacts_shaped_force(env: ManagerBasedRLEnv,
               desired_contact, # 计划足端相位差,待实现
               gait_force_sigma, # 待定义系数
               sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="*FOOT"),
               ) -> torch.Tensor: 
    sensor:ContactSensor = env.scene[sensor_cfg.name]
    foot_forces = torch.norm(sensor.data.net_forces_w[:, asset_cfg.body_ids, :], dim = -1)
    foot_reward = -(1 - desired_contact) * (1 - torch.exp(-1 * foot_forces[:, :] ** 2 / gait_force_sigma))
    reward = torch.sum(foot_reward, dim=-1)
    return reward

def reward_tracking_contacts_shaped_vel(env: ManagerBasedRLEnv, 
        desired_contact,
        gait_vel_sigma,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="*FOOT"),):
    asset:RigidObject = env.scene[asset_cfg.name]
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim = -1)
    foot_reward = -desired_contact * (1 - torch.exp(-1 * foot_velocities[:, :] ** 2 / gait_vel_sigma))
    reward = torch.sum(foot_reward, dim=-1) / 4
    return reward