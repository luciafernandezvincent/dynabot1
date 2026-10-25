"""Nucleo del modelo de actuador serie-elastico, sin dependencias de Isaac Lab.

Se separa del actuador para poder testear la integracion numerica sin levantar Isaac Sim.

Modelo
------
Entre el motor y el eslabon hay un resorte, en vez de una union rigida::

    objetivo --PD--> [ motor, J_m ] --resorte (k, c)--> [ pata (PhysX) ]

- El PD actua sobre el MOTOR, no sobre la pata (es lo que hace un servo real: su lazo se
  cierra contra su propio encoder, que esta del lado del motor).
- La pata recibe unicamente el torque del resorte, que es el que efectivamente se transmite
  por la transmision.
- El estado del motor (posicion y velocidad) se integra aca, fuera de PhysX, porque el rotor
  no interactua con contactos: solo con su resorte.

De esta estructura salen las dos cosas que el modelo rigido no puede reproducir: el modo
oscilatorio motor-pata (el "rebote") y la diferencia entre angulo de motor y angulo de pata.
"""

from __future__ import annotations

import math

import torch


def resolve_per_joint(
    value: float | int | dict, joint_names: list[str], num_envs: int, device, name: str = "parameter"
) -> torch.Tensor:
    """Expande un escalar o un dict {regex: valor} a un tensor (num_envs, num_joints).

    Replica la semantica de los cfg de Isaac Lab sin depender de sus helpers internos, que
    cambiaron de nombre y de firma entre versiones (``_parse_joint_parameter`` paso a ser
    ``resolve_joint_parameter``). Con esto el actuador funciona igual en cualquier version.
    """
    import re

    num_joints = len(joint_names)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return torch.full((num_envs, num_joints), float(value), device=device)

    if not isinstance(value, dict):
        raise TypeError(
            f"{name}: se esperaba un numero o un dict {{regex: valor}}, se recibio "
            f"{type(value).__name__}. Si quedo sin setear en el cfg, hay que darle un valor."
        )

    out = torch.full((num_envs, num_joints), float("nan"), device=device)
    for pattern, pattern_value in value.items():
        matched = False
        for index, joint_name in enumerate(joint_names):
            if re.fullmatch(pattern, joint_name):
                out[:, index] = float(pattern_value)
                matched = True
        if not matched:
            raise ValueError(f"{name}: el patron '{pattern}' no coincide con ninguna junta de {joint_names}")

    missing = torch.isnan(out[0]).nonzero().flatten().tolist()
    if missing:
        raise ValueError(
            f"{name}: quedaron juntas sin valor: {[joint_names[i] for i in missing]}"
        )
    return out


def natural_frequency_hz(spring_stiffness: float, rotor_inertia: float) -> float:
    """Frecuencia natural del modo motor-resorte, en Hz: f = sqrt(k/J) / 2*pi."""
    if rotor_inertia <= 0.0:
        raise ValueError(f"rotor_inertia must be positive, got {rotor_inertia}")
    return math.sqrt(spring_stiffness / rotor_inertia) / (2.0 * math.pi)


def recommended_substeps(spring_stiffness: float, rotor_inertia: float, dt: float,
                         motor_stiffness: float = 0.0, max_omega_dt: float = 0.2) -> int:
    """Cuantos subpasos hacen falta para integrar el rotor con precision a paso `dt`.

    Sobre el rotor actuan DOS rigideces en paralelo: la del resorte (contra la pata) y la del
    PD del servo (contra el objetivo), asi que la frecuencia relevante es
    ``sqrt((k + kp) / J)`` y no solo la del resorte. Omitir ``motor_stiffness`` subestima los
    subpasos necesarios.

    El criterio de ESTABILIDAD del Euler semi-implicito sobre un oscilador es omega*dt < 2;
    para PRECISION conviene omega*dt <= 0.2, que es el valor por defecto.

    El amortiguamiento no entra en el criterio porque ``integrate_series_elastic`` lo integra
    de forma implicita, lo cual es incondicionalmente estable. Si se integrara explicitamente
    haria falta ademas ``(c + kd) * dt / J < 2``, que con un rotor liviano es la restriccion
    dominante y la que en la practica produce NaN.
    """
    omega = math.sqrt((spring_stiffness + motor_stiffness) / rotor_inertia)
    return max(1, math.ceil(omega * dt / max_omega_dt))


def integrate_series_elastic(
    motor_pos: torch.Tensor,
    motor_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    motor_stiffness: torch.Tensor,
    motor_damping: torch.Tensor,
    spring_stiffness: torch.Tensor,
    spring_damping: torch.Tensor,
    rotor_inertia: torch.Tensor,
    dt: float,
    substeps: int = 1,
    motor_effort_limit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Avanza el estado del motor un paso de fisica y devuelve el torque del resorte.

    `joint_pos` y `joint_vel` se mantienen constantes durante los subpasos: son el estado que
    reporta PhysX y no avanza hasta el proximo `sim.step()`.

    Todos los tensores son de forma (num_envs, num_joints), salvo `dt` y `substeps`.

    Returns:
        (motor_pos, motor_vel, spring_torque), el torque a aplicar sobre la junta de PhysX.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    dt_sub = dt / substeps

    for _ in range(substeps):
        # --- torques que NO dependen de la velocidad nueva del rotor (parte explicita)
        spring_stiff_torque = spring_stiffness * (motor_pos - joint_pos)
        motor_stiff_torque = motor_stiffness * (target_pos - motor_pos)

        # El amortiguamiento se integra de forma IMPLICITA: su coeficiente multiplica a la
        # velocidad NUEVA en vez de a la actual. Con un rotor liviano, integrarlo explicitamente
        # exige (c + kd) * dt / J < 2, que es la restriccion que mas facil se viola.
        explicit_torque = motor_stiff_torque - spring_stiff_torque + spring_damping * joint_vel
        damping_coeff = spring_damping + motor_damping
        explicit_torque = explicit_torque + motor_damping * target_vel

        if motor_effort_limit is not None:
            # Donde el motor satura, su torque pasa a ser una constante: deja de aportar
            # amortiguamiento y su termino va entero a la parte explicita.
            motor_torque_raw = motor_stiff_torque + motor_damping * (target_vel - motor_vel)
            saturated = motor_torque_raw.abs() > motor_effort_limit
            if saturated.any():
                clamped = torch.sign(motor_torque_raw) * motor_effort_limit
                explicit_torque = torch.where(
                    saturated,
                    clamped - spring_stiff_torque + spring_damping * joint_vel,
                    explicit_torque,
                )
                damping_coeff = torch.where(saturated, spring_damping, damping_coeff)

        # Euler semi-implicito con amortiguamiento implicito:
        #   v_new = (v + dt * tau_explicit / J) / (1 + dt * c / J)
        motor_vel = (motor_vel + dt_sub * explicit_torque / rotor_inertia) / (
            1.0 + dt_sub * damping_coeff / rotor_inertia
        )
        motor_pos = motor_pos + motor_vel * dt_sub

    spring_torque = spring_stiffness * (motor_pos - joint_pos) + spring_damping * (motor_vel - joint_vel)
    return motor_pos, motor_vel, spring_torque
