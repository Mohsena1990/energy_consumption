#!/usr/bin/env python
"""
Run Full Pipeline
=================
Execute all pipeline scripts in sequence.

Usage:
    python scripts/run_pipeline.py [--config CONFIG_PATH] [--steps STEPS]

Examples:
    # Run full pipeline
    python scripts/run_pipeline.py

    # Run only specific steps
    python scripts/run_pipeline.py --steps 0 1 2

    # Use custom config
    python scripts/run_pipeline.py --config configs/custom.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


SCRIPTS = [
    ("00_make_dataset.py", "Load and prepare data"),
    ("01_run_fs.py", "Run feature selection"),
    ("02_eval_fs_shap_mcda.py", "Evaluate FS with SHAP + MCDA"),
    ("03_optimize_models.py", "Optimize model hyperparameters"),
    ("04_evaluate_and_safeguards.py", "Evaluate models and safeguards"),
    ("05_select_best_model.py", "Select best model with MCDA"),
    ("06_interpret_champion.py", "Interpret champion model"),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run full CO2 forecasting pipeline')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                       help='Steps to run (0-6). Default: all')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID to use')
    return parser.parse_args()


def run_script(script_name: str, config: str = None, run_id: str = None) -> int:
    """Run a single script."""
    scripts_dir = Path(__file__).parent
    script_path = scripts_dir / script_name

    cmd = [sys.executable, str(script_path)]

    if config:
        cmd.extend(['--config', config])

    if run_id:
        cmd.extend(['--run-id', run_id])

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: {script_name} failed with return code {result.returncode}")
        return result.returncode

    return 0


def main():
    args = parse_args()

    print("=" * 60)
    print("CO2 Forecasting Pipeline")
    print("=" * 60)

    # Determine which steps to run
    if args.steps is not None:
        steps_to_run = args.steps
    else:
        steps_to_run = list(range(len(SCRIPTS)))

    print(f"\nSteps to run: {steps_to_run}")
    print(f"Config: {args.config or 'default'}")
    print(f"Run ID: {args.run_id or 'auto-generated'}")

    # Run scripts — use a single run_id for all steps
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M")

    for step in steps_to_run:
        if step < 0 or step >= len(SCRIPTS):
            print(f"Warning: Step {step} is invalid, skipping")
            continue

        script_name, description = SCRIPTS[step]

        print(f"\nStep {step}: {description}")

        result = run_script(script_name, args.config, run_id)

        if result != 0:
            print(f"\nPipeline stopped at step {step}")
            return result

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nCheck outputs/ directory for results.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
