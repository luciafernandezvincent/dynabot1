"""Actuador serie-elastico para Isaac Lab: modela la elasticidad entre motor y eslabon.

Motivacion
----------
En el modelo rigido el angulo de motor y el angulo de pata son el mismo numero, y la politica
se entrena observando ese valor. En el robot real el encoder esta del lado del MOTOR y entre el
motor y la pata hay una transmision que cede, con lo cual los dos angulos difieren y aparece un
modo oscilatorio que el simulador rigido no puede reproducir.

Este actuador introduce ese modo::

    objetivo --PD--> [ motor, J_m ] --resorte (k, c)--> [ pata (PhysX) ]

El PD se cierra contra el angulo del MOTOR (como el lazo interno de un servo real) y a la junta
de PhysX se le aplica unicamente el torque del resorte. El estado del motor se integra fuera de
PhysX porque el rotor solo interactua con su resorte, no con contactos.

El angulo de motor queda accesible para la observacion via ``mdp.motor_pos_rel``, de modo que la
red reciba la misma magnitud que recibe en el robot real.

Calibracion
-----------
``spring_stiffness`` y ``spring_damping`` NO son las ganancias del PD: son propiedades mecanicas
de la transmision y hay que medirlas. El ensayo mas directo es desplazar una pata a mano y
soltarla ("pluck test"): del periodo de la oscilacion libre sale ``k = J * omega_n^2`` y de la
caida de la envolvente sale el amortiguamiento. Mientras no esten medidas, los valores por
defecto son una estimacion y el modelo reproducira un modo a la frecuencia equivocada.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DCMotorCfg, IdealPDActuator
from isaaclab.utils import configclass

from .series_elastic_core import (
    integrate_series_elastic,
    natural_frequency_hz,
    recommended_substeps,
    resolve_per_joint,
)

if TYPE_CHECKING:
    from isaaclab.utils.types import ArticulationActions


class SeriesElasticActuator(IdealPDActuator):
    """PD sobre el motor, resorte entre motor y eslabon.

    Hereda de :class:`IdealPDActuator` para reutilizar ``stiffness`` y ``damping``, que aca pasan
    a ser las ganancias del lazo del SERVO (lado motor) y no un PD sobre la pata.
    """

    cfg: SeriesElasticActuatorCfg

    def __init__(self, cfg: SeriesElasticActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        if cfg.dt is MISSING or cfg.dt is None:
            raise ValueError(
                "SeriesElasticActuatorCfg.dt es obligatorio y debe valer lo mismo que sim.dt: el "
                "actuador integra el estado del motor con ese paso. Con sim.dt = 0.005, poner dt=0.005."
            )
        self._dt = float(cfg.dt)

        def resolve(value, name):
            if value is MISSING:
                raise ValueError(f"SeriesElasticActuatorCfg.{name} es obligatorio y quedo sin setear")
            return resolve_per_joint(value, self.joint_names, self._num_envs, self._device, name)

        # propiedades mecanicas de la transmision, por junta
        self.spring_stiffness = resolve(cfg.spring_stiffness, "spring_stiffness")
        self.spring_damping = resolve(cfg.spring_damping, "spring_damping")
        self.rotor_inertia = resolve(cfg.rotor_inertia, "rotor_inertia")
        if (self.rotor_inertia <= 0.0).any():
            raise ValueError("rotor_inertia debe ser positiva en todas las juntas")

        # Ganancias del lazo del servo. Se resuelven desde el cfg y NO desde self.stiffness: ese
        # atributo no lo define ActuatorBase y su existencia depende de la version de Isaac Lab.
        self.motor_stiffness = resolve(cfg.stiffness, "stiffness")
        self.motor_damping = resolve(cfg.damping, "damping")

        # Limite de torque del motor. Se usa el que resolvio la clase base si esta disponible.
        limit = getattr(self, "actuator_effort_limit", None)
        if not isinstance(limit, torch.Tensor):
            limit = resolve(cfg.effort_limit if cfg.effort_limit is not MISSING else 1.0e9, "effort_limit")
        self._motor_effort_limit = limit

        # Curva par-velocidad del motor (DCMotorCfg). Se evalua sobre la velocidad del ROTOR
        # dentro del integrador, no sobre la del eslabon: con un resorte en el medio, los dos
        # giran distinto y el que trabaja contra su propia curva es el motor.
        self._motor_saturation_effort = (
            None if cfg.saturation_effort is MISSING or cfg.saturation_effort is None
            else resolve(cfg.saturation_effort, "saturation_effort")
        )
        self._motor_velocity_limit = (
            None if self._motor_saturation_effort is None
            else resolve(cfg.velocity_limit, "velocity_limit")
        )
        if self._motor_velocity_limit is not None and (self._motor_velocity_limit <= 0.0).any():
            raise ValueError(
                "velocity_limit debe ser positivo en todas las juntas cuando se usa "
                "saturation_effort: es el divisor de la curva par-velocidad."
            )

        self._transmitted_effort_limit = (
            None if cfg.transmitted_effort_limit is None
            else resolve(cfg.transmitted_effort_limit, "transmitted_effort_limit")
        )

        # estado del motor; se inicializa con el estado de la junta en el primer compute
        self.motor_pos = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.motor_vel = torch.zeros_like(self.motor_pos)
        self._initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        # ultimo angulo de pata visto, para poder reportar la deflexion del resorte
        self._joint_pos = torch.zeros_like(self.motor_pos)

        self._substeps = self._resolve_substeps(cfg.substeps)

    # ------------------------------------------------------------------ setup

    def _resolve_substeps(self, configured: int | None) -> int:
        """Elige los subpasos de integracion y avisa si el valor pedido es insuficiente.

        Sobre el rotor actuan en paralelo la rigidez del resorte y la del PD, de modo que la
        frecuencia a resolver es sqrt((k + kp) / J). Se toma el caso mas exigente entre juntas.
        """
        worst_stiffness = float(self.spring_stiffness.max().item())
        worst_motor_stiffness = float(self.motor_stiffness.max().item())
        lightest_rotor = float(self.rotor_inertia.min().item())
        needed = recommended_substeps(worst_stiffness, lightest_rotor, self._dt, worst_motor_stiffness)

        if configured is None:
            print(
                f"[INFO][SeriesElasticActuator] subpasos = {needed} (automatico). Modo del resorte: "
                f"{natural_frequency_hz(worst_stiffness, lightest_rotor):.1f} Hz con k="
                f"{worst_stiffness:.1f} y J={lightest_rotor:.5f}."
            )
            return needed

        configured = int(configured)
        if configured < needed:
            omega_dt = math.sqrt((worst_stiffness + worst_motor_stiffness) / lightest_rotor) * self._dt
            print(
                f"[WARNING][SeriesElasticActuator] substeps={configured} es bajo: con estos "
                f"parametros omega*dt = {omega_dt:.2f} y harian falta {needed} subpasos para "
                f"integrar el modo con precision. Puede dar oscilaciones espurias o NaN."
            )
        return configured

    # ------------------------------------------------------------------ API

    def reset(self, env_ids: Sequence[int]):
        """Marca esos entornos para que el estado del motor se re-sincronice con la junta.

        Sin esto el rotor arrancaria el episodio nuevo con la deflexion del anterior, lo que
        inyecta un torque de resorte espurio en el primer paso.
        """
        super().reset(env_ids)
        if isinstance(env_ids, slice):
            self._initialized[env_ids] = False
        else:
            self._initialized[torch.as_tensor(env_ids, device=self._device, dtype=torch.long)] = False

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # sincronizar el rotor con la junta en los entornos recien reiniciados: sin deflexion
        if not bool(self._initialized.all()):
            fresh = ~self._initialized
            self.motor_pos[fresh] = joint_pos[fresh]
            self.motor_vel[fresh] = joint_vel[fresh]
            self._initialized[fresh] = True

        self._joint_pos = joint_pos.clone()
        target_pos = control_action.joint_positions
        target_vel = control_action.joint_velocities
        if target_pos is None:
            target_pos = self.motor_pos
        if target_vel is None:
            target_vel = torch.zeros_like(self.motor_vel)

        self.motor_pos, self.motor_vel, spring_torque = integrate_series_elastic(
            motor_pos=self.motor_pos,
            motor_vel=self.motor_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            motor_stiffness=self.motor_stiffness,
            motor_damping=self.motor_damping,
            spring_stiffness=self.spring_stiffness,
            spring_damping=self.spring_damping,
            rotor_inertia=self.rotor_inertia,
            dt=self._dt,
            substeps=self._substeps,
            motor_effort_limit=self._motor_effort_limit,
            motor_saturation_effort=self._motor_saturation_effort,
            motor_velocity_limit=self._motor_velocity_limit,
        )

        # A la junta de PhysX solo le llega lo que transmite el resorte.
        #
        # NO se acota al limite de torque del motor: en un actuador serie-elastico el resorte
        # puede transmitir transitoriamente MAS que el motor, porque devuelve energia almacenada.
        # Acotarlo ahi suprimiria justamente los picos de impacto que este modelo existe para
        # reproducir. El limite del motor ya se aplico adentro, sobre el torque del motor.
        # `transmitted_effort_limit` queda como red de seguridad numerica, apagada por defecto.
        feedforward = control_action.joint_efforts
        self.computed_effort = spring_torque + (feedforward if feedforward is not None else 0.0)
        if self._transmitted_effort_limit is None:
            self.applied_effort = self.computed_effort
        else:
            self.applied_effort = torch.clamp(
                self.computed_effort, -self._transmitted_effort_limit, self._transmitted_effort_limit
            )

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    # ------------------------------------------------------------------ inspeccion

    @property
    def spring_deflection(self) -> torch.Tensor:
        """Angulo de motor menos angulo de pata: lo que el modelo rigido fuerza a cero."""
        return self.motor_pos - self._joint_pos


@configclass
class SeriesElasticActuatorCfg(DCMotorCfg):
    """Configuracion del actuador serie-elastico.

    ``stiffness`` y ``damping`` (heredados) son las ganancias del lazo del SERVO, que actuan sobre
    el angulo del motor. ``spring_stiffness`` y ``spring_damping`` son las propiedades mecanicas de
    la transmision, y hay que medirlas: ver la nota de calibracion del modulo.

    Hereda de :class:`DCMotorCfg` y no de ``IdealPDActuatorCfg`` para conservar
    ``saturation_effort``, o sea la curva par-velocidad del motor: el techo de par baja a medida
    que el rotor gira mas rapido, en vez de ser una constante. Importa para este robot, donde toda
    la investigacion de torque se midio contra esa curva (effort_limit = saturation_effort = 9 Nm).
    Dejarlo sin setear (``MISSING``) vuelve al techo plano ``effort_limit``.
    """

    class_type: type = SeriesElasticActuator

    dt: float = 1/200
    """Paso de integracion del estado del motor. Debe coincidir con ``sim.dt``."""

    spring_stiffness: dict[str, float] | float = 80.0
    """Rigidez de la transmision, en Nm/rad. Es mecanica, no una ganancia de control."""

    spring_damping: dict[str, float] | float = 0.15
    """Amortiguamiento de la transmision, en Nm*s/rad. Con 0 el modo no se apaga nunca."""

    rotor_inertia: dict[str, float] | float = 3e-5
    """Inercia del rotor reflejada a la salida, en kg*m^2. Define la frecuencia del modo."""

    substeps: int | None = None
    """Subpasos de integracion por paso de fisica. Con ``None`` se elige automaticamente."""

    transmitted_effort_limit: dict[str, float] | float | None = None
    """Tope al torque que el resorte transmite a la junta, en Nm. ``None`` = sin tope.

    Es una red de seguridad numerica, no fisica: el resorte PUEDE transmitir mas que el motor
    (devuelve energia almacenada), asi que acotarlo al limite del motor suprimiria los picos de
    impacto. Usarlo solo si la simulacion diverge.
    """
