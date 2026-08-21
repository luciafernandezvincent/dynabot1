#!/usr/bin/env python3
"""Run parameter sweep using autoresearch."""

import json
import itertools
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

def generate_experiments(sweep_config):
    """Generate all experiment combinations from sweep config."""
    # Get sweep parameters
    sweeps = sweep_config["sweeps"]
    base = sweep_config["base_experiment"]

    # Generate all combinations
    param_names = list(sweeps.keys())
    param_values = list(sweeps.values())

    experiments = []
    for values in itertools.product(*param_values):
        exp_params = dict(zip(param_names, values))
        experiments.append({
            "params": exp_params,
            "base": base
        })

    return experiments


def create_experiment_yaml(params):
    """Create YAML config file for experiment with given params."""
    config = {
        "env": {
            "rewards": {
                "feet_air_time": {"weight": params.get("feet_air_time_weight", 0.22)},
                "action_rate_l2": {"weight": params.get("action_rate_weight", -0.1)},
                "lin_vel_z_l2": {"weight": params.get("lin_vel_z_weight", -2.0)},
            }
        },
        "agent": {
            "algorithm": {
                "learning_rate": params.get("learning_rate", 1.0e-3),
            }
        }
    }

    return config


def run_sweep():
    """Run complete parameter sweep."""
    # Load sweep config
    with open("scripts/rsl_rl/experiments/sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)

    # Generate all experiments
    experiments = generate_experiments(sweep_config)
    print(f"[INFO] Generated {len(experiments)} experiments")
    print(f"[INFO] Sweeping over: {list(sweep_config['sweeps'].keys())}\n")

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i, exp in enumerate(experiments):
        params = exp["params"]

        # Create experiment name
        exp_name = f"sweep_{i:03d}_lr{params['learning_rate']:.0e}_air{params['feet_air_time_weight']}"
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp_name}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")

        # Create YAML config
        yaml_config = create_experiment_yaml(params)

        # Save YAML
        yaml_path = f"scripts/rsl_rl/experiment_configs/sweep_{i:03d}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)

        # Run training
        cmd = [
            "python", "scripts/rsl_rl/train_delay.py",
            f"--task={exp['base']['task']}",
            "--headless",
            "--num_envs=4096",
            f"--name={exp_name}",
            f"--experiment_config={yaml_path}"
        ]

        print(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Experiment {i+1} completed successfully")
            results[exp_name] = {
                "status": "SUCCESS",
                "params": params
            }
        else:
            print(f"✗ Experiment {i+1} FAILED")
            print(result.stderr[-500:])  # Last 500 chars of error
            results[exp_name] = {
                "status": "FAILED",
                "params": params,
                "error": result.stderr[-500:]
            }

    # Save sweep results
    results_file = Path("scripts/rsl_rl/experiments/sweep_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Sweep results saved to: {results_file}")
    print(f"\nCompleted: {sum(1 for r in results.values() if r['status'] == 'SUCCESS')}/{len(experiments)}")


if __name__ == "__main__":
    print("[INFO] Starting AutoResearch parameter sweep...")
    run_sweep()
