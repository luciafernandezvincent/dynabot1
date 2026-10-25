"""Observaciones propias para la tarea de locomocion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from dynabot1.actuators import SeriesElasticActuator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _motor_state(asset: Articulation, pick_velocity: bool) -> torch.Tensor:
    """Angulo (o velocidad) del lado del MOTOR para todas las juntas.

    Las juntas accionadas por un :class:`SeriesElasticActuator` reportan el estado del rotor; el
    resto cae al estado de la junta, que es lo correcto cuando esa junta no modela elasticidad
    (ahi motor y eslabon coinciden por definicion).
    """
    source = asset.data.joint_vel if pick_velocity else asset.data.joint_pos
    out = source.clone()
    for actuator in asset.actuators.values():
        if isinstance(actuator, SeriesElasticActuator):
            value = actuator.motor_vel if pick_velocity else actuator.motor_pos
            out[:, actuator.joint_indices] = value
    return out


def motor_pos_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angulo de MOTOR relativo a la postura nominal.

    Equivalente a ``mdp.joint_pos_rel`` pero del lado del motor, que es lo que mide el encoder
    del servo en el robot real. Con actuadores rigidos devuelve exactamente lo mismo que
    ``joint_pos_rel``, asi que se puede dejar puesta sin cambiar nada hasta activar la elasticidad.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return _motor_state(asset, pick_velocity=False) - asset.data.default_joint_pos


def motor_vel_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Velocidad del lado del MOTOR, relativa a la velocidad nominal."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _motor_state(asset, pick_velocity=True) - asset.data.default_joint_vel


def spring_deflection(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Deflexion de la transmision (angulo de motor menos angulo de pata), por junta.

    No esta pensada para la politica -- en el robot real no se mide sin instrumentacion extra --
    sino para diagnostico: es la magnitud que el modelo rigido fuerza a cero, y la que permite
    verificar que el modelo elastico se este comportando como se espera.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.zeros_like(asset.data.joint_pos)
    for actuator in asset.actuators.values():
        if isinstance(actuator, SeriesElasticActuator):
            out[:, actuator.joint_indices] = actuator.spring_deflection
    return out
