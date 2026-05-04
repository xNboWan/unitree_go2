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
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

   
def reward_tracking_contacts_shaped_force(
        env: ManagerBasedRLEnv,
        desired_contact, # 计划足端相位差,待实现
        gait_force_sigma, # 待定义系数
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT"),
        ) -> torch.Tensor: 
    contact_sensor:ContactSensor = env.scene[sensor_cfg.name]
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim = -1)
    foot_reward = (1 - desired_contact) * (1 - torch.exp(-1 * foot_forces ** 2 / gait_force_sigma))
    reward = torch.sum(foot_reward, dim=-1)
    return reward

def reward_tracking_contacts_shaped_vel(
        env: ManagerBasedRLEnv, 
        desired_contact,     # 计划足端相位差,待实现
        gait_vel_sigma,     # 待定义系数
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT"))-> torch.Tensor:
    asset:Articulation = env.scene[asset_cfg.name]
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim = -1)
    foot_reward = desired_contact * (1 - torch.exp(-1 * foot_velocities ** 2 / gait_vel_sigma))
    reward = torch.sum(foot_reward, dim=-1) / 4
    return reward

def reward_action_smoothness_1(
        env: ManagerBasedRLEnv,
) -> torch.Tensor:
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    return torch.sum(diff, dim=1)

class action_smoothness_2_term(ManagerTermBase):
   
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: torch.Tensor):
        self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        diff = torch.square(action - 2.0 * prev_action + self.prev_prev_action)
        reward = torch.sum(diff, dim=1)
        self.prev_prev_action[:] = prev_action.clone()
        return reward

def reward_feet_slip(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset:Articulation = env.scene[asset_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.
    last_contact = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids, 2] > 1.
    contact_filt = torch.logical_or(contact, last_contact)
    foot_velocities = torch.square(torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0:2], dim=-1))
    rew_slip = torch.sum(contact_filt.float() * foot_velocities, dim = 1)
    return rew_slip

def reward_feet_contact_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT")
):
    reference_heights = 0
    asset: Articulation = env.scene[asset_cfg.name]
    near_ground = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - reference_heights < 0.03
    foot_velocities = torch.square(torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim = 2))
    rew_contact_vel = torch.sum(near_ground.float() * foot_velocities, dim = 1)
    return rew_contact_vel

def reward_feet_clearance_cmd_linear(
        env: ManagerBasedRLEnv,
        foot_phases, # 足端相位差,范围0-1, 待实现
        target_height, # 目标位置, 待实现
        desired_contact,  # 计划足端相位差,待实现
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT")
):
    phases = 1 - torch.abs(1.0 - torch.clip((foot_phases * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    asset:Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # whw实现参考 target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - desired_contact)
    return torch.sum(rew_foot_clearance, dim = 1)

def _reward_feet_clearance_cmd_linear(self):
    phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
    target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
    return torch.sum(rew_foot_clearance, dim=1)    

class reward_feet_impact_vel_term(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_foot_vel_z = None

    def reset(self, env_ids: torch.Tensor):
        if self.prev_foot_vel_z is not None:
            self.prev_foot_vel_z[env_ids] = 0.0

    def __call__(self, 
                 env: ManagerBasedRLEnv,
                 sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT")) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        current_foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        if self.prev_foot_vel_z is None:
            self.prev_foot_vel_z = torch.zeros_like(current_foot_vel_z)
        contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        contact_states = torch.norm(contact_forces, dim=-1) > 1.0
        prev_downward_vel = torch.clip(self.prev_foot_vel_z, -100.0, 0.0)
        rew_foot_impact_vel = contact_states.float() * torch.square(prev_downward_vel)
        self.prev_foot_vel_z[:] = current_foot_vel_z.clone()
        return torch.sum(rew_foot_impact_vel, dim=1)
    
def _reward_feet_impact_vel(self):
    prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
    contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

    rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

    return torch.sum(rew_foot_impact_vel, dim=1)

def reward_orientation_control(
        env:ManagerBasedRLEnv,
):
    '''
    待实现
    '''
    return

def _reward_orientation_control(self):
        # Penalize non flat base orientation
    roll_pitch_commands = self.env.commands[:, 10:12]
    quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
    quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

    desired_base_quat = quat_mul(quat_roll, quat_pitch)
    desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

    return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

def reward_raibert_heuristic(
        env:ManagerBasedRLEnv,
):
    '''
    待实现
    '''
    return

def _reward_raibert_heuristic(self):
    cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                          cur_footsteps_translated[:, i, :])

    # nominal positions: [FR, FL, RR, RL]
    if self.env.cfg.commands.num_commands >= 13:
        desired_stance_width = self.env.commands[:, 12:13]
        desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
    else:
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

    if self.env.cfg.commands.num_commands >= 14:
        desired_stance_length = self.env.commands[:, 13:14]
        desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
    else:
        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
    phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = self.env.commands[:, 4]
    x_vel_des = self.env.commands[:, 0:1]
    yaw_vel_des = self.env.commands[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

    reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

    return reward