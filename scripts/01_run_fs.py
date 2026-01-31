#!/usr/bin/env python
"""
Script 01: Run Feature Selection
================================
Run all three feature selection strategies: linear, nonlinear, and consensus.

Usage:
    python scripts/01_run_fs.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/fs_linear_scores.csv
    - outputs/runs/<run_id>/tables/fs_nonlinear_scores.csv
    - outputs/runs/<run_id>/tables/fs_consensus_summary.csv
    - outputs/runs/<run_id>/models/fs_results.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy, load_pickle
)
from src.data_io import load_processed_data
from src.splits import load_cv_plan
from src.fs import run_all_fs_options


def parse_args():
    parser = argparse.ArgumentParser(description='Run feature selection')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (uses existing or creates new)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.run_id:
        config.run_id = args.run_id

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)

    logger.info("=" * 60)
    logger.info("Script 01: Run Feature Selection")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading processed data...")
    processed_dir = Path('data/processed')

    X = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    logger.info(f"Loaded: {len(X)} samples, {len(X.columns)} features")

    # =========================================
    # Run All Feature Selection Options
    # =========================================
    logger.info("-" * 40)
    logger.info("Running all feature selection methods...")

    fs_results = run_all_fs_options(X, y, cv_plan, config)

    # Save results
    save_json_numpy(fs_results, dirs['models'] / 'fs_results.json')

    # Save individual FS scores
    for fs_name, results in fs_results.items():
        # Save selected features
        save_json_numpy(
            {'features': results['selected_features']},
            dirs['tables'] / f'{fs_name}.json'
        )

        # Save detailed scores
        if 'steps' in results:
            for step in results['steps']:
                if 'scores' in step and step['scores']:
                    scores_df = pd.DataFrame(step['scores'])
                    scores_df.to_csv(
                        dirs['tables'] / f"{fs_name}_{step['name']}_scores.csv",
                        index=False
                    )

    # Summary table
    summary_rows = []
    for fs_name, results in fs_results.items():
        summary_rows.append({
            'fs_option': fs_name,
            'n_features': results['n_selected'],
            'features': ', '.join(results['selected_features'][:10]) + ('...' if len(results['selected_features']) > 10 else '')
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs['tables'] / 'fs_summary.csv', index=False)

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Feature Selection Complete!")
    for fs_name, results in fs_results.items():
        logger.info(f"  {fs_name}: {results['n_selected']} features")
    logger.info(f"Results saved to: {dirs['tables']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
