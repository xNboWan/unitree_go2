# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from .unitree_go2_rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # episode_length
        self.episode_length_s = 20

        # commends change
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # --task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 追踪xy平面线速度

        self.rewards.track_ang_vel_z_exp.weight = 0.5   # 追踪z方向角速度

        # -- penalties
        self.rewards.lin_vel_z_l2.weight = -2.0         # 惩罚z方向线速度
        self.rewards.ang_vel_xy_l2.weight = -0.05       # 惩罚xy平面角速度 
        self.rewards.dof_torques_l2.weight = -1.0e-5    # 惩罚关节转动
        self.rewards.dof_acc_l2.weight = -2.5e-7        # 惩罚关节加速度
        self.rewards.action_rate_l2.weight = -0.25      # 惩罚动作变化率
        
        self.rewards.feet_air_time.weight = 0.01        # 奖励腾空时间
        # self.rewards.undesired_contacts.weight = -1.0

        # --optional penalties
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_pos_limits.weight = 0.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
        # no terrain curriculum
        self.curriculum.terrain_levels = None



class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
