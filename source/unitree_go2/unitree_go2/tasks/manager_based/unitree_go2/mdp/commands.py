import torch
from isaaclab.managers import CommandTerm
from .commands_cfg import WalkTheseWaysCommandCfg
class WalkTheseWaysCommand(CommandTerm):

    cfg: WalkTheseWaysCommandCfg  

    def __init__(self, cfg: WalkTheseWaysCommandCfg, env):

        super().__init__(cfg, env)
        
        # 全局时钟
        self.gait_indices = self.cfg.gait_indices

        self._command = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | None = None) -> None:

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        min_vel, max_vel = self.vel_range
        new_commands = (max_vel - min_vel) * torch.rand(len(env_ids), 1, device=self.device) + min_vel
        
        self._command[env_ids] = new_commands

    def compute(self, dt: float) -> None:
        
        delta_t = self.gait_indices * dt
        self.gait_indices += delta_t % 1.0
        pass