from isaaclab.utils import configclass

from .rough_env_cfg import AnymalDRoughEnvCfg
from dynabot1.assets.dynabot import DYNABOT_1_CFG  

@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AnymalDFlatEnvCfg_PLAY(AnymalDFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

class GraphArtResEnvCfg(AnymalDFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 1. Modificamos articulation_props para cambiar el fix_root_link
        articulation_props_modificado = DYNABOT_1_CFG.spawn.articulation_props.replace(
            fix_root_link=True
        )
        
        # 2. Metemos esas propiedades modificadas dentro del spawn del robot
        self.scene.robot = DYNABOT_1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=DYNABOT_1_CFG.spawn.replace(
                articulation_props=articulation_props_modificado
            )
        )
        
