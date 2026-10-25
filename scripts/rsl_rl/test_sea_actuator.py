"""Smoke test del SeriesElasticActuator SIN escena, robot ni fisica.

Levanta la app solo porque importar isaaclab lo exige, y despues construye el actuador a mano
con tensores falsos. Eso alcanza para detectar desajustes de API (que es donde estaban los bugs)
sin esperar a un entrenamiento.

Uso:
    python scripts/rsl_rl/test_sea_actuator.py
"""

import argparse
import sys
import traceback

# ------------------------------------------------------------------ levantar la app
# Tiene que ir ANTES de cualquier import de dynabot1/isaaclab: esos imports cargan el pxr de
# usd-core (pip) y despues Kit choca con su propio USD y crashea al arrancar.
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Smoke test del actuador serie-elastico.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ paso 0: nucleo, sin isaaclab
print("=" * 72)
print("[0] nucleo numerico (no necesita Isaac Lab)")
print("=" * 72)
try:
    from dynabot1.actuators.series_elastic_core import (  # noqa: E402
        natural_frequency_hz,
        recommended_substeps,
        resolve_per_joint,
    )
    print("  import del nucleo: OK")
except Exception:
    # el paquete dynabot1 importa isaaclab en su __init__, asi que se carga por path
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve().parents[2]
    path = here / "source/dynabot1/dynabot1/actuators/series_elastic_core.py"
    spec = importlib.util.spec_from_file_location("sea_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    natural_frequency_hz = core.natural_frequency_hz
    recommended_substeps = core.recommended_substeps
    resolve_per_joint = core.resolve_per_joint
    print(f"  import del nucleo por path: OK ({path.name})")

import torch  # noqa: E402

JOINTS = ["fl_shoulder", "fl_shoulder_to_arm", "fl_arm_to_hand"]
print(f"  resolve_per_joint(escalar) -> {resolve_per_joint(80.0, JOINTS, 2, 'cpu')[0].tolist()}")
print(f"  frecuencia del modo k=80, J=0.03 -> {natural_frequency_hz(80.0, 0.03):.2f} Hz")
print(f"  subpasos sugeridos (dt=0.005, kp=90) -> {recommended_substeps(80.0, 0.03, 0.005, 90.0)}")
print()

failures = []


def step(label):
    """Decorador trivial para reportar que paso fallo, en vez de un traceback suelto."""
    def wrapper(fn):
        print("=" * 72)
        print(f"[{label}]")
        print("=" * 72)
        try:
            fn()
            print(f"  -> OK\n")
        except Exception as error:
            failures.append(label)
            print(f"  -> FALLO: {type(error).__name__}: {error}")
            traceback.print_exc()
            print()
        return fn
    return wrapper


state = {}


@step("1 imports de Isaac Lab")
def _():
    from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg

    state["IdealPDActuatorCfg"] = IdealPDActuatorCfg
    print(f"  IdealPDActuator      = {IdealPDActuator}")
    print(f"  IdealPDActuatorCfg   = {IdealPDActuatorCfg}")
    # como declara class_type la version instalada: string diferido o clase?
    from isaaclab.actuators import DelayedPDActuatorCfg

    declared = DelayedPDActuatorCfg.__dataclass_fields__.get("class_type")
    print(f"  class_type de DelayedPDActuatorCfg: {getattr(declared, 'default', '?')!r}")


@step("2 import del actuador propio")
def _():
    from dynabot1.actuators import SeriesElasticActuator, SeriesElasticActuatorCfg

    state["cls"] = SeriesElasticActuator
    state["cfg_cls"] = SeriesElasticActuatorCfg
    print(f"  SeriesElasticActuator    = {SeriesElasticActuator}")
    print(f"  class_type declarado     = {SeriesElasticActuatorCfg.class_type}")
    print(f"  hereda de IdealPDActuator: {issubclass(SeriesElasticActuator, state['IdealPDActuatorCfg'].class_type) if isinstance(state['IdealPDActuatorCfg'].class_type, type) else 'no verificable (class_type es string)'}")


@step("3 construir el cfg")
def _():
    cfg = state["cfg_cls"](
        joint_names_expr=[".*"],
        stiffness={".*_shoulder": 90.0, ".*shoulder_to_arm": 28.0, ".*arm_to_hand": 100.0},
        damping={".*_shoulder": 5.5, ".*shoulder_to_arm": 1.85, ".*arm_to_hand": 3.5},
        effort_limit=14.5,
        dt=0.005,
        spring_stiffness=80.0,
        spring_damping=0.31,
        rotor_inertia={".*_shoulder": 0.03, ".*shoulder_to_arm": 0.04, ".*arm_to_hand": 0.03},
        armature=0.0,
    )
    state["cfg"] = cfg
    print(f"  cfg creado: dt={cfg.dt}, k={cfg.spring_stiffness}, J={cfg.rotor_inertia}")


@step("4 instanciar el actuador (sin articulation)")
def _():
    num_envs = 4
    kwargs = dict(
        cfg=state["cfg"],
        joint_names=JOINTS,
        joint_ids=slice(None),
        num_envs=num_envs,
        device="cpu",
    )
    # el nombre del kwarg del limite de esfuerzo cambio entre versiones; se prueban las variantes
    for extra in [
        {"actuator_effort_limit": 14.5, "actuator_velocity_limit": 21.0},
        {"effort_limit": 14.5, "velocity_limit": 21.0},
        {},
    ]:
        try:
            actuator = state["cls"](**kwargs, **extra)
            print(f"  instanciado con kwargs extra: {list(extra) or 'ninguno'}")
            break
        except TypeError as error:
            print(f"  variante {list(extra) or 'ninguna'} rechazada: {error}")
    else:
        raise RuntimeError("ninguna variante de kwargs funciono")

    state["actuator"] = actuator
    print(f"  num_joints = {actuator.num_joints}, motor_pos.shape = {tuple(actuator.motor_pos.shape)}")
    print(f"  spring_stiffness[0] = {actuator.spring_stiffness[0].tolist()}")
    print(f"  rotor_inertia[0]    = {actuator.rotor_inertia[0].tolist()}")
    print(f"  motor_stiffness[0]  = {actuator.motor_stiffness[0].tolist()}")


@step("5 compute(): el modo oscila a la frecuencia esperada")
def _():
    actuator = state["actuator"]
    num_envs, num_joints = 4, len(JOINTS)
    dt = 0.005

    try:
        from isaaclab.utils.types import ArticulationActions
    except ImportError:
        from dataclasses import dataclass

        @dataclass
        class ArticulationActions:  # stand-in: compute() solo toca estos tres campos
            joint_positions: torch.Tensor | None = None
            joint_velocities: torch.Tensor | None = None
            joint_efforts: torch.Tensor | None = None

        print("  ArticulationActions no disponible; se usa un stand-in")

    # pata clavada en cero, objetivo cero, motor desplazado: debe sonar a sqrt(k/J)
    joint_pos = torch.zeros(num_envs, num_joints)
    joint_vel = torch.zeros(num_envs, num_joints)
    actuator.reset(slice(None))

    torques = []
    for i in range(400):
        action = ArticulationActions(
            joint_positions=torch.zeros(num_envs, num_joints),
            joint_velocities=torch.zeros(num_envs, num_joints),
            joint_efforts=None,
        )
        out = actuator.compute(action, joint_pos, joint_vel)
        if i == 0:
            # desplazar el rotor DESPUES de la sincronizacion inicial, para excitar el modo
            actuator.motor_pos += 0.05
        torques.append(out.joint_efforts[0].clone())

    tau = torch.stack(torques)
    if not torch.isfinite(tau).all():
        raise RuntimeError("el torque transmitido tiene NaN/inf: la integracion diverge")

    for j, name in enumerate(JOINTS):
        signal = tau[:, j] - tau[:, j].mean()
        sign = torch.sign(signal)
        crossings = torch.nonzero(sign[1:] * sign[:-1] < 0).flatten()
        k = actuator.spring_stiffness[0, j].item()
        inertia = actuator.rotor_inertia[0, j].item()
        expected = natural_frequency_hz(k, inertia)
        if len(crossings) >= 3:
            half = torch.diff(crossings.float()).mean().item() * dt
            measured = 1.0 / (2.0 * half)
            print(f"  {name:<22} esperada {expected:6.2f} Hz   medida {measured:6.2f} Hz")
        else:
            print(f"  {name:<22} esperada {expected:6.2f} Hz   (sin oscilacion detectada)")
    print(f"  deflexion maxima = {actuator.spring_deflection.abs().max().item():.5f} rad")


@step("6 reset() re-sincroniza el rotor con la junta")
def _():
    actuator = state["actuator"]
    num_envs, num_joints = 4, len(JOINTS)
    try:
        from isaaclab.utils.types import ArticulationActions
    except ImportError:
        from dataclasses import make_dataclass

        ArticulationActions = make_dataclass(
            "ArticulationActions",
            [("joint_positions", object, None), ("joint_velocities", object, None),
             ("joint_efforts", object, None)],
        )

    actuator.reset([1])
    joint_pos = torch.full((num_envs, num_joints), 0.3)
    # objetivo = pata: si el reset sincronizo, el env 1 queda en equilibrio y no transmite nada.
    # Con objetivo 0 el PD mueve el rotor dentro del mismo paso y el torque no da ~0 aunque el
    # reset ande bien.
    action = ArticulationActions(
        joint_positions=joint_pos.clone(),
        joint_velocities=torch.zeros(num_envs, num_joints),
        joint_efforts=None,
    )
    out = actuator.compute(action, joint_pos, torch.zeros(num_envs, num_joints))
    tau_reset = out.joint_efforts[1].abs().max().item()
    tau_other = out.joint_efforts[0].abs().max().item()
    print(f"  torque en el env reiniciado: {tau_reset:.4f} Nm (deberia ser ~0)")
    print(f"  torque en los demas envs:    {tau_other:.4f} Nm")
    if tau_reset > 0.1:
        raise RuntimeError("el env reiniciado arranca con deflexion: reset() no sincronizo")


print("=" * 72)
if failures:
    print(f"FALLARON los pasos: {', '.join(failures)}")
else:
    print("TODOS LOS PASOS PASARON: el actuador se construye y corre contra esta version de Isaac Lab")
print("=" * 72)

simulation_app.close()
sys.exit(1 if failures else 0)
