#!/usr/bin/env python3
"""Run dynabot1 evaluation experiments using autoresearch."""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add autoresearch to path
try:
    from autoresearch import ResultsTable
except ImportError:
    print("[ERROR] autoresearch not installed. Run: pip install git+https://github.com/karpathy/autoresearch.git")
    sys.exit(1)


def run_experiments():
    """Run all evaluation experiments."""
    experiments = {
        "baseline": {
            "cmd": "python scripts/rsl_rl/eval.py --task=Dyna1-Flat-v0 --load_run=baseline_test --num_steps=5000",
            "name": "Baseline Configuration"
        },
        "smooth_gait": {
            "cmd": "python scripts/rsl_rl/eval.py --task=Dyna1-Flat-v0 --load_run=smooth_gait --num_steps=5000",
            "name": "Smooth Gait"
        },
        "slower_gait": {
            "cmd": "python scripts/rsl_rl/eval.py --task=Dyna1-Flat-v0 --load_run=slower_gait --num_steps=5000",
            "name": "Slower Gait"
        },
    }

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n{'='*80}")
    print(f"AutoResearch Experiment Suite - {timestamp}")
    print(f"{'='*80}\n")

    for exp_id, exp_config in experiments.items():
        print(f"\n[{exp_id}] {exp_config['name']}")
        print(f"Command: {exp_config['cmd']}")
        print("-" * 80)

        # Run experiment
        result = subprocess.run(exp_config["cmd"], shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Experiment failed!")
            print(result.stderr)
            results[exp_id] = {"status": "FAILED", "error": result.stderr}
        else:
            print(f"[SUCCESS] Experiment completed")
            results[exp_id] = {"status": "SUCCESS"}

    return results


def compare_results():
    """Compare results from all experiments using ResultsTable."""
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80 + "\n")

    # Create a results table
    table = ResultsTable()

    # Look for result files in logs
    log_dirs = list(Path("logs/rsl_rl").glob("*/eval/results.json"))

    if not log_dirs:
        print("[WARNING] No eval results found. Run evaluations first.")
        return

    all_metrics = {}

    for result_file in log_dirs:
        with open(result_file) as f:
            data = json.load(f)
            exp_name = result_file.parent.parent.name
            all_metrics[exp_name] = data

            # Add to table
            table.add_row(
                experiment=exp_name,
                smoothness=data.get("movement_smoothness", 0),
                orientation_stability=data.get("orientation_stability_0to1", 0),
                velocity_tracking=data.get("velocity_tracking_accuracy_0to1", 0),
                fall_rate=data.get("fall_rate_per_episode", 0),
                impact_force=data.get("impact_force_mean", 0),
            )

    # Print table
    print(table)

    # Save detailed comparison
    comparison_file = Path("scripts/rsl_rl/experiments/results_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ Detailed results saved to: {comparison_file}")

    # Print summary
    print("\n" + "="*80)
    print("BEST PERFORMERS:")
    print("="*80)

    metrics = ["movement_smoothness", "orientation_stability_0to1", "velocity_tracking_accuracy_0to1"]

    for metric in metrics:
        best_exp = max(all_metrics.items(), key=lambda x: x[1].get(metric, 0))
        print(f"  {metric}: {best_exp[0]} ({best_exp[1].get(metric, 0):.4f})")


def main():
    """Main execution."""
    print("[INFO] Starting AutoResearch evaluation suite...\n")

    # Run experiments
    run_experiments()

    # Compare results
    compare_results()

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
