# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort:skip

##
# Scene definition
##

class Go2SceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # robot
    unitree_go2 = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/unitree_go2")

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # 【核心修复】：位置控制的 scale 必须改小，通常 0.25 弧度（约14度）足以应对平地行走
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="unitree_go2", 
        joint_names=[ 
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ], 
        use_default_offset=True, # 基于默认站立姿态进行位置偏移控制
        scale=0.25 
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("unitree_go2")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    add_go2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("unitree_go2", body_names=["base"]), "mass_distribution_params": (0.1, 0.5), "operation": "add"},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("unitree_go2", joint_names=[".*"]),
            "position_range": (-0.5, 0.5), # 缩小一点随机范围，方便位置控制起步
            "velocity_range": (-0.1, 0.1),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 1. 生存与终止
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # 2. 核心任务目标 (稍微调高一点奖励，明确目标)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go2")}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("unitree_go2")}
    )

    # 3. 惩罚项 (【大幅削减权重】，让狗敢于迈腿)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    base_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    
    # 力矩和速度惩罚缩小 10 倍！
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.00002, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.00001, params={"asset_cfg": SceneEntityCfg("unitree_go2")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"asset_cfg": SceneEntityCfg("unitree_go2", body_names=["base"]), "minimum_height": 0.2}
    )

@configclass
class CommandsCfg:
    """Velocity commands for the robot to track."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="unitree_go2",
        resampling_time_range=(10.0, 10.0), 
        rel_standing_envs=0.05, # 加大一点静止环境的比例，防止它只会跑不会站
        rel_heading_envs=1.0, 
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi)
        ),
    )

@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 environment."""
    scene = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()

    episode_length_s = 20 # 建议20秒

    def __post_init__(self):
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        self.decimation = 4  
        self.sim.dt = 0.005