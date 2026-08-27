from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.imu import ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import dynabot1.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    imu_head = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        # offset=ImuCfg.OffsetCfg(pos=(0.3, 0.0, 0.0)),  # Posición más adelante (forward)
        # accel_range=(300.0, 300.0),
        # gyro_range=(2000.0, 2000.0),
        update_period=0.01,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            #lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 1.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(mean=0.0, std=0.02))
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Gnoise(mean=0.039, std=0.0078),
        )
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        projected_gravity =  ObsTerm(
            func=mdp.base_ang_vel,
            noise=Gnoise(mean=0.2003, std=0.0023),
        )
        #projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Gnoise(mean=0.0, std=0.048))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.1))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0,std=0.1))
        actions = ObsTerm(func=mdp.last_action)

        # height_scan = ObsTerm(
        #    func=mdp.height_scan,
        #    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #    noise=Unoise(n_min=-0.1, n_max=0.1),
        #    clip=(-1.0, 1.0),
        #)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.25, 1.5), #(-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0), # raro
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)  # 0.1
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        #weight=0.125,
        weight=0.22,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*hand_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*arm_link"), "threshold": 1.0},
    )
    # peso 0 por defecto: no cambia el comportamiento de la baseline, solo lo deja disponible
    # para tunear via YAML (env.rewards.foot_clearance.weight / .params.target_height).
    # feet_air_time solo premia la DURACION sin contacto, no la altura: un pie puede "levantarse"
    # unos milimetros y satisfacer igual el umbral de tiempo. Este termino sí mide altura.
    # tambien en peso 0: disponibles para tunear via YAML sin alterar la baseline.
    # gait fuerza el patron de trote (diagonales sincronizadas); air_time_variance castiga que
    # unas patas trabajen mas que otras (en exp_001 la delantera izq despegaba 4x mas que la
    # trasera der). Ambos adaptados de la config de Spot de Isaac Lab.
    gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (
                ("front_left_hand_link", "back_right_hand_link"),
                ("front_right_hand_link", "back_left_hand_link"),
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*hand_link")},
    )
    # peso 0 por defecto: disponible para tunear via YAML. Pedido del usuario (27/08): premia que
    # el tiempo de apoyo de cada pata se acerque a target_contact_time, para marcha mas uniforme y
    # mas lenta (target mas alto que el duty factor actual). air_time_variance (arriba) no logro
    # esto en exp_042/058 -- este apunta a un VALOR objetivo, no solo a reducir la varianza.
    stance_time = RewTerm(
        func=mdp.stance_time_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*hand_link"),
            "target_contact_time": 0.15,
            "std": 0.01,
        },
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*hand_link"),
            "target_height": 0.05,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    # peso 0 por defecto: disponible para tunear via YAML. Recompensa mantener el segmento medio
    # de la pata (arm_link, la "rodilla") por encima de min_height (score.py: termino
    # "knee_clearance"). min_height=0.11 es el objetivo real pedido por el usuario (26/08), no el
    # 0.0579 de baseline_ar (esa es solo la referencia de comparacion en score.py:
    # REF_KNEE_CLEARANCE_MIN, que a proposito NO cambia con esto). v2 (exp(-error/std), pesado por
    # velocidad del segmento): mejor resultado medido hasta ahora (exp_040). Una v3 (filtro por
    # direccion del comando) salio peor (exp_041) y se descarto, ver rewards.py.
    knee_clearance = RewTerm(
        func=mdp.knee_clearance_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*arm_link"),
            "min_height": 0.11,
            "std": 0.005,
            "tanh_mult": 2.0,
        },
    )
    # peso 0: disponible para tunear via YAML. Penaliza desviarse de la pose default. Motivo
    # (usuario, 21/08/2026): en el robot REAL los actuadores hacen mucha mejor fuerza cerca de la
    # pose default; una postura muy flexionada exige mas torque para el mismo trabajo y fuerza los
    # servos. dof_pos_limits no sirve para esto: solo castiga cerca del LIMITE de la junta, no la
    # distancia a la default.
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)

    # variante quirurgica: penaliza desviarse de la default SOLO en las juntas de hombro
    # (base_to_*_shoulder), que son las que definen si el codo va plegado hacia adentro. Deja
    # libres shoulder_to_arm y arm_to_hand, que son las que hacen el arco del paso. Motivo:
    # penalizar TODAS las juntas (exp_016/017) endereza la postura pero achica el paso -despeje
    # 40.6 -> 10.3 mm, swing 197 -> 134 ms, zancada 2.56 -> 4.54 Hz: pasitos apurados y rasantes.
    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder"])},
    )

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0e-4)#-1.0e-5)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base_link"), "threshold": 1.0},
    )
    shoulder_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*shoulder_link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False