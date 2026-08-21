#!/usr/bin/env python3
"""AutoResearch experiment runner for dynabot1 evaluation."""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

# Define experiments to run
EXPERIMENTS = [
    {
        "name": "baseline",
        "task": "Dyna1-Flat-v0",
        "load_run": "baseline_test",
        "num_steps": 5000,
    },
    {
        "name": "smooth_gait",
        "task": "Dyna1-Flat-v0",
        "load_run": "smooth_gait",
        "num_steps": 5000,
    },
    {
        "name": "slower_gait",
        "task": "Dyna1-Flat-v0",
        "load_run": "slower_gait",
        "num_steps": 5000,
    },
]


def run_experiment(exp_config: dict):
    """Run a single evaluation experiment."""
    cmd = [
        "python",
        "scripts/rsl_rl/eval.py",
        f"--task={exp_config['task']}",
        f"--load_run={exp_config['load_run']}",
        f"--num_steps={exp_config['num_steps']}",
    ]

    print(f"\n{'='*80}")
    print(f"Running: {exp_config['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def compare_results():
    """Compare results from all experiments."""
    results_dir = Path("logs/rsl_rl")
    all_results = {}

    for exp in EXPERIMENTS:
        # Find latest eval results
        results_path = results_dir / exp["task"].replace(":", "_") / "eval" / "results.json"

        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
                all_results[exp["name"]] = results
                print(f"\n{exp['name']}:")
                print(json.dumps(results, indent=2))

    # Save comparison
    comparison_file = Path("scripts/rsl_rl/experiments/results_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {comparison_file}")
    return all_results


def main():
    """Run all experiments and compare results."""
    print("[INFO] Starting AutoResearch experiment suite...")

    success_count = 0
    for exp in EXPERIMENTS:
        if run_experiment(exp):
            success_count += 1
        else:
            print(f"[ERROR] Experiment '{exp['name']}' failed")

    print(f"\n[INFO] Completed {success_count}/{len(EXPERIMENTS)} experiments")

    # Compare results
    compare_results()


if __name__ == "__main__":
    main()
