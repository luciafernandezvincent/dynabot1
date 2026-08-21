"""Aplica overrides de env/agent desde un YAML de experimento.

Compartido por train_delay.py, eval.py y play.py: el mismo YAML resuelto que se uso para
entrenar tiene que aplicarse tambien al evaluar y al grabar el video, porque eval.py y play.py
reconstruyen agent_cfg desde el default de la tarea (no leen el checkpoint para saber su forma).
Sin esto, un override que cambia la ARQUITECTURA de la red (no un peso de reward ni un
hiperparametro de PPO) entrena bien pero despues eval.py/play.py fallan al cargar el checkpoint
en un modelo con la forma por defecto (visto en exp_008_bigger_net, 21/08/2026).
"""

import logging

import yaml

logger = logging.getLogger(__name__)


def apply_overrides(cfg_obj, overrides: dict, path: str = "") -> None:
    """Recursively apply a nested dict of overrides onto a config object's attributes.

    - A leaf value (not a dict) is set directly with setattr.
    - A dict value whose current attribute is a plain dict (e.g. a reward term's ``params``) is merged
      key-by-key, recursing into any nested config object found inside that dict (e.g. ``params.sensor_cfg``)
      instead of overwriting it.
    - A dict value whose current attribute is a config object (has ``__dict__``) recurses into it.
    - ``None`` disables the attribute (e.g. an optional reward/event term), matching the pattern used
      throughout the env cfgs (``self.rewards.undesired_contacts = None``).
    """
    for key, value in overrides.items():
        full_path = f"{path}.{key}" if path else key
        if not hasattr(cfg_obj, key):
            logger.warning(f"[WARNING] Config field '{full_path}' not found, skipping.")
            continue
        current = getattr(cfg_obj, key)

        if value is None:
            setattr(cfg_obj, key, None)
            logger.info(f"[INFO] Set '{full_path}' = None")
        elif isinstance(value, dict) and isinstance(current, dict):
            for sub_key, sub_value in value.items():
                nested_cfg = current.get(sub_key)
                if isinstance(sub_value, dict) and hasattr(nested_cfg, "__dict__"):
                    apply_overrides(nested_cfg, sub_value, f"{full_path}.{sub_key}")
                else:
                    current[sub_key] = sub_value
                    logger.info(f"[INFO] Set '{full_path}.{sub_key}' = {sub_value}")
        elif isinstance(value, dict) and hasattr(current, "__dict__"):
            apply_overrides(current, value, full_path)
        else:
            setattr(cfg_obj, key, value)
            logger.info(f"[INFO] Set '{full_path}' = {value}")


def apply_experiment_config(env_cfg, agent_cfg, config_path: str) -> None:
    """Load a YAML file with 'env:' and/or 'agent:' sections and apply them as config overrides."""
    with open(config_path) as f:
        experiment_cfg = yaml.safe_load(f) or {}

    if "env" in experiment_cfg:
        apply_overrides(env_cfg, experiment_cfg["env"], path="env")
    if "agent" in experiment_cfg:
        apply_overrides(agent_cfg, experiment_cfg["agent"], path="agent")
