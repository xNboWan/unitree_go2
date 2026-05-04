from isaaclab.managers import CommandTermCfg
import configclass
from .commands import WalkTheseWaysCommand

@configclass
class WalkTheseWaysCommandCfg(CommandTermCfg):   
    class_type: type = WalkTheseWaysCommand

    gait_indices: float = 0.0

