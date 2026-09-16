from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def foot_impact_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penaliza pisar fuerte: la fuerza de contacto del pie EN EL INSTANTE del apoyo (touchdown).

    Pedido del usuario (08/09/2026): "que no haga movimientos con mucha fuerza en la pisada". Hasta
    ahora ningun reward miraba la fuerza de contacto -- `undesired_contacts` solo cuenta contactos
    del arm_link, y `dof_torques_l2` mide el torque de los actuadores, que no es lo mismo que el
    golpe contra el piso.

    Se mide igual que el juez: `eval.py` calcula `impact_force_mean` como la norma de
    `net_forces_w` sobre los pies en los pasos donde `compute_first_contact` es verdadero. Esta
    penalizacion usa exactamente esa cantidad, asi que optimizarla mueve la metrica que se puntua.

    Forma: solo se cobra el EXCESO por encima de `threshold` (`clamp(F - threshold, min=0)`), asi
    que apoyar suave no cuesta nada y el gradiente aparece unicamente cuando la pisada es dura. Es
    lineal y no cuadratica a proposito: con fuerzas de ~100 N un termino cuadratico daria valores
    de 1e4 que aplastarian al resto de los rewards.

    Devuelve la suma sobre las cuatro patas, en newtons. Se usa con peso NEGATIVO y chico
    (del orden de -0.005 a -0.02): con un exceso tipico de ~50 N por pisada, -0.01 aporta -0.5 en
    el paso del apoyo, comparable a los terminos de tarea.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    excess = torch.clamp(forces - threshold, min=0.0)
    return torch.sum(excess * first_contact.float(), dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    A duration-based air-time reward (ver ``feet_air_time``) no le da al agente ninguna senal
    sobre cuanto se levanta el pie: alcanza con satisfacer un tiempo minimo sin contacto, sin
    importar la altura. Este termino, adaptado de la config de Spot en Isaac Lab, penaliza el
    error respecto de una altura objetivo, pesado por la velocidad lateral del pie (para que solo
    aplique durante el swing, no en apoyo).
    """
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def knee_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Recompensa que el segmento medio de la pata (``arm_link``, la "rodilla") se mantenga por
    encima de una altura minima sobre el piso, mientras se esta moviendo horizontalmente.

    Medido con eval.py (26/08): yendo hacia atras el arm_link puede pasar a pocos cm del piso
    (casi "gateando") sin llegar a disparar el sensor de contacto de ``undesired_contacts``.

    Historial (26/08): v1 (exp_036/037) penalty x^2 sin acotar, empeoraba al subir el peso. v2
    (esta, exp_038/039/040) exp(-error/std) pesado por velocidad del segmento, mismo patron que
    ``foot_clearance_reward`` -- con std bien calibrado (0.005) mejora knee_height_min_backward_m
    y es el mejor resultado medido hasta ahora (exp_040, combinado con otros factores). Se probo
    una v3 filtrando por direccion del comando en vez de por velocidad del segmento (hipotesis:
    concentrar la señal solo en el caso "para atras"), pero salio PEOR que v2 aislada (exp_041:
    0.4820 vs exp_039: 0.5284) -- se descarto y se volvio a v2.
    """
    asset = env.scene[asset_cfg.name]
    knee_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    violation_sq = torch.square(torch.clamp(min_height - knee_z, min=0.0))
    knee_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = violation_sq * knee_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground.

    Adaptado de la config de Spot en Isaac Lab. Sirve para corregir marchas asimetricas: medido en
    exp_001, la delantera izquierda despegaba 71 mm y la trasera derecha 19 mm (casi 4x). Ninguna
    otra metrica ni reward penalizaba esa diferencia.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def stance_time_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, target_contact_time: float, std: float
) -> torch.Tensor:
    """Recompensa que el tiempo de apoyo (contacto) de cada pata se acerque a un valor objetivo.

    Pedido del usuario (27/08): un termino que mida el tiempo en el piso de la pata para que el
    movimiento sea mas uniforme entre patas (todas tienden al mismo tiempo de apoyo, en vez de solo
    penalizar la varianza como ``air_time_variance_penalty``, que en exp_042/058 no funciono) y
    mas lento (target_contact_time mayor que el duty factor actual alarga la zancada). Mismo
    patron que ``foot_clearance_reward``: ``exp(-error/std)`` acotado en (0, 1].
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    error = torch.square(contact_time - target_contact_time)
    return torch.exp(-torch.sum(error, dim=1) / std)


class GaitReward(ManagerTermBase):
    """Premia una marcha con pares de patas sincronizados (trote) para cuadrupedos.

    Copiado de la config de Spot en Isaac Lab. Penaliza la diferencia de timing de contacto entre
    los pares indicados en ``synced_feet_pair_names``: fuerza a que las diagonales pisen juntas y
    a que un par este en el aire mientras el otro apoya. Es lo que le da forma de marcha en vez de
    cuatro patas moviendose cada una por su cuenta.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        # para patas sincronizadas, los tiempos de aire/contacto deben coincidir
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # para patas asincronicas, el contacto de una debe coincidir con el aire de la otra
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # solo aplica si el comando de velocidad supera el umbral
        cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


class BaseAngularAcceleration(ManagerTermBase):
    """Penaliza la ACELERACION angular de la base: el tiron del cuerpo, no su inclinacion.

    Motivo (16/09/2026): `orientation_smoothness` es el unico termino del score que ningun reward
    mira, y se nota -- quedo entre 0.49 y 0.577 en las 11 corridas de la tanda sin_delay, sin
    importar que se cambiara. Los dos terminos de orientacion que existen apuntan a otra cosa:

      - `flat_orientation_l2` (-5.0) penaliza el ANGULO: que el cuerpo este inclinado.
      - `ang_vel_xy_l2` (-0.05) penaliza la VELOCIDAD angular: que el cuerpo rote.
      - este penaliza la ACELERACION angular: que el giro cambie de golpe.

    Un cuerpo puede estar perfectamente horizontal (bien para el primero), girando despacio (bien
    para el segundo) y llegar igual con sacudones. Eso es lo que sentiria un IMU montado en el lomo
    y es lo que el juez puntua.

    REPLICA DE LA METRICA DEL JUEZ. `compute_orientation_smoothness` en `scripts/rsl_rl/eval.py`
    hace: euler(quat) -> unwrap -> primera diferencia / dt -> segunda diferencia / dt -> media del
    VALOR ABSOLUTO sobre los tres ejes. Tres decisiones se copian de ahi, no son gusto:

      1. **L1, no L2.** El juez promedia |acc|. Un termino cuadratico castigaria los picos mucho
         mas que la media y optimizaria otra cosa.
      2. **Los tres ejes, yaw incluido.** El juez no excluye el yaw (a diferencia de
         `orientation_stability`, que solo mira roll/pitch). Penalizar la ACELERACION de yaw no
         castiga girar: un giro a tasa constante tiene aceleracion cero. Castiga cambiar de tasa.
      3. **La derivada se toma de la velocidad angular del MUNDO** (`root_ang_vel_w`), no de los
         angulos de Euler reconstruidos paso a paso. Las tasas de Euler y la velocidad angular del
         mundo difieren por una matriz cinematica que tiende a la identidad cuanto mas horizontal
         esta el cuerpo; con roll/pitch de ~0.03 rad (lo medido en esta tanda) la diferencia es de
         segundo orden, y evita tener que guardar dos estados previos y desenrollar angulos dentro
         del reward.

    GATE POR COMANDO. Se cobra solo con comando de velocidad por encima de `velocity_threshold`
    (mismo patron que `GaitReward`). Sin eso el optimo del termino es quedarse quieto, que esta
    medido en este repo: una politica sin entrenar, con las cuatro patas apoyadas el 99.5% del
    tiempo, saca mejor estabilidad y suavidad que cualquier marcha entrenada. Un reward "puro" de
    estabilidad SIN gate paga por no caminar.

    ESCALA. El juez reporta `1/(1+mean|acc|)`, y de ahi sale la magnitud cruda: la baseline de la
    tanda (smoothness 0.1363) tiene mean|acc| = 6.3 rad/s^2 por eje, o sea ~19 sumando los tres.
    Con peso -5e-3 eso aporta del orden de -0.15 al reward de episodio, comparable a
    `joint_deviation` (-0.15) y `foot_impact` (-0.18). Peso 0 por default: no cambia ninguna
    baseline hasta que un YAML lo active. Calibrar con un smoke test de 5 iteraciones y mirar
    `Episode_Reward/base_ang_acc` antes de gastar un entrenamiento entero -- la leccion de
    `torque_rate` (exp_119), que corrio con un peso 4x corto y aporto menos del 1% del reward.

    QUE VIGILAR. La aceleracion angular baja por dos vias: caminando mas suave o caminando menos.
    Los indicadores son `stride_frequency_hz_mean` y `step_length_m`: si la frecuencia baja y el
    paso se acorta a la vez, el termino compro quietud y no suavidad.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.prev_ang_vel = torch.zeros((env.num_envs, 3), device=env.device)
        # arranca en False: el primer paso de un episodio no tiene velocidad previa valida
        self.has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev_ang_vel[env_ids] = 0.0
        # sin esto, el salto entre la velocidad angular del ultimo paso de un episodio y la del
        # primero del siguiente contaria como un sacudon enorme. Es el reset, no la marcha: el juez
        # lo descarta con `valid_acc` y aca se descarta igual.
        self.has_prev[env_ids] = False

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, velocity_threshold: float
    ) -> torch.Tensor:
        ang_vel = self.asset.data.root_ang_vel_w
        acc = torch.sum(torch.abs(ang_vel - self.prev_ang_vel) / env.step_dt, dim=1)
        acc = torch.where(self.has_prev, acc, torch.zeros_like(acc))
        self.prev_ang_vel[:] = ang_vel
        self.has_prev[:] = True
        cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
        return torch.where(cmd > velocity_threshold, acc, torch.zeros_like(acc))
