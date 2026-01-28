#!/usr/bin/env python
"""
Script 03: Optimize Models
==========================
Train and optimize all forecasting models using the selected feature set.
Uses PSO/GWO swarm optimization with walk-forward CV.

Usage:
    python scripts/03_optimize_models.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/models/<model>_best_params.json
    - outputs/runs/<run_id>/models/<model>_model.pkl
    - outputs/runs/<run_id>/tables/optimization_summary.csv
    - outputs/runs/<run_id>/figures/optimization_<model>.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy, load_json, get_latest_run_id
)
from src.data_io import load_processed_data
from src.splits import load_cv_plan
from src.optimization import optimize_all_models, train_optimized_model
from src.models import ModelRegistry
from src.reporting import plot_optimization_history, set_plot_style


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize forecasting models')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to optimize (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.run_id:
        config.run_id = args.run_id
    else:
        latest = get_latest_run_id(config.output.base_dir)
        if latest:
            config.run_id = latest

    if args.models:
        config.model.models = args.models

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    set_plot_style()

    logger.info("=" * 60)
    logger.info("Script 03: Optimize Models")
    logger.info("=" * 60)
    logger.info(f"Models to optimize: {config.model.models}")
    logger.info(f"Optimizer: {config.optimization.optimizer}")

    # Load data
    logger.info("Loading data...")
    processed_dir = Path('data/processed')

    X_full = load_processed_data(processed_dir / 'X_full.parquet')
    y = pd.read_parquet(processed_dir / 'y.parquet')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    # Load selected features
    selected_fs_path = dirs['tables'] / 'selected_feature_set.json'
    if not selected_fs_path.exists():
        logger.error("Selected feature set not found. Run 02_eval_fs_shap_mcda.py first.")
        return 1

    selected_fs = load_json(selected_fs_path)
    selected_features = selected_fs['selected_features']

    # Filter to selected features
    available_features = [f for f in selected_features if f in X_full.columns]
    X = X_full[available_features]

    logger.info(f"Using {len(available_features)} features from {selected_fs['selected_fs_option']}")

    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Drop NaN
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")

    # =========================================
    # Optimize All Models
    # =========================================
    logger.info("-" * 40)
    logger.info("Running swarm optimization for all models...")

    optimization_results = optimize_all_models(
        X, y, cv_plan, config,
        output_dir=dirs['models']
    )

    # Save optimization summary
    summary_rows = []
    for model_name, results in optimization_results.items():
        if 'error' in results:
            summary_rows.append({
                'model': model_name,
                'status': 'failed',
                'error': results['error']
            })
        else:
            summary_rows.append({
                'model': model_name,
                'status': 'success',
                'best_fitness': results.get('best_fitness'),
                'best_params': str(results.get('best_params', {}))
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs['tables'] / 'optimization_summary.csv', index=False)

    # =========================================
    # Generate Optimization Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating optimization plots...")

    for model_name, results in optimization_results.items():
        if 'history' in results and results['history']:
            plot_optimization_history(
                results['history'],
                title=f"Optimization History - {model_name.upper()}",
                output_path=dirs['figures'] / f'optimization_{model_name}.png'
            )

    # =========================================
    # Train Final Models with Best Params
    # =========================================
    logger.info("-" * 40)
    logger.info("Training final models with optimized parameters...")

    trained_models = {}
    for model_name, results in optimization_results.items():
        if 'best_params' not in results:
            continue

        try:
            model = train_optimized_model(X, y, model_name, results['best_params'])
            model.save(dirs['models'] / f'{model_name}_model.pkl')
            trained_models[model_name] = model
            logger.info(f"  {model_name}: Trained and saved")
        except Exception as e:
            logger.error(f"  {model_name}: Training failed - {e}")

    # Save best params summary
    best_params_summary = {}
    for model_name, results in optimization_results.items():
        if 'best_params' in results:
            best_params_summary[model_name] = {
                'params': results['best_params'],
                'fitness': results.get('best_fitness')
            }
    save_json_numpy(best_params_summary, dirs['tables'] / 'best_params_all_models.json')

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Model Optimization Complete!")
    logger.info(f"  Models optimized: {len(trained_models)}")
    for model_name, results in optimization_results.items():
        if 'best_fitness' in results:
            logger.info(f"    {model_name}: fitness = {results['best_fitness']:.4f}")
    logger.info(f"  Results saved to: {dirs['models']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
