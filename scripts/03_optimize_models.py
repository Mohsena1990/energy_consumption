#!/usr/bin/env python
"""
Script 03: Optimize Models
==========================
Train and optimize all forecasting models using top-N feature sets from MCDA.
Trains models on each FS option for side-by-side comparison.

Usage:
    python scripts/03_optimize_models.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/models/<fs_option>/<model>_model.pkl
    - outputs/runs/<run_id>/tables/optimization_summary.csv
    - outputs/runs/<run_id>/tables/fs_comparison_summary.csv
    - outputs/runs/<run_id>/figures/optimization_<fs_option>_<model>.png
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
    parser.add_argument('--fs-option', type=str, default=None,
                       help='Specific FS option to use (default: all top-N)')
    return parser.parse_args()


def optimize_for_fs_option(
    X_full: pd.DataFrame,
    y: pd.Series,
    cv_plan,
    config: Config,
    fs_option: str,
    selected_features: list,
    output_dir: Path,
    logger
) -> dict:
    """Optimize all models for a specific FS option."""
    logger.info(f"Optimizing models for FS option: {fs_option}")
    logger.info(f"  Features ({len(selected_features)}): {selected_features[:5]}...")

    # Create FS-specific output directory
    fs_output_dir = output_dir / fs_option
    fs_output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to selected features
    available_features = [f for f in selected_features if f in X_full.columns]
    X = X_full[available_features]

    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y_aligned = y.loc[common_idx]

    # Drop NaN
    valid_mask = ~(X.isnull().any(axis=1) | y_aligned.isnull())
    X = X[valid_mask]
    y_aligned = y_aligned[valid_mask]

    logger.info(f"  Training data: {len(X)} samples, {len(available_features)} features")

    # Optimize all models
    optimization_results = optimize_all_models(
        X, y_aligned, cv_plan, config,
        output_dir=fs_output_dir
    )

    # Train final models with best params
    trained_models = {}
    for model_name, results in optimization_results.items():
        if 'best_params' not in results:
            continue
        try:
            model = train_optimized_model(X, y_aligned, model_name, results['best_params'])
            model.save(fs_output_dir / f'{model_name}_model.pkl')
            trained_models[model_name] = model
            fitness = results.get('best_fitness', 0)
            logger.info(f"    {model_name}: Trained (fitness={fitness:.4f})")
        except Exception as e:
            logger.error(f"    {model_name}: Training failed - {e}")

    # Save best params for this FS option
    best_params_summary = {}
    for model_name, results in optimization_results.items():
        if 'best_params' in results:
            best_params_summary[model_name] = {
                'params': results['best_params'],
                'fitness': results.get('best_fitness')
            }
    save_json_numpy(best_params_summary, fs_output_dir / 'best_params_all_models.json')

    return {
        'fs_option': fs_option,
        'n_features': len(available_features),
        'features': available_features,
        'optimization_results': optimization_results,
        'trained_models': list(trained_models.keys()),
        'output_dir': str(fs_output_dir)
    }


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
    logger.info("Script 03: Optimize Models (Multi-FS Support)")
    logger.info("=" * 60)
    logger.info(f"Models to optimize: {config.model.models}")
    logger.info(f"Optimizer: {config.optimization.optimizer}")

    # Load data
    logger.info("Loading data...")
    processed_dir = Path('data/processed')

    X_full = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    # Load selected features info (with top-N options)
    selected_fs_path = dirs['tables'] / 'selected_feature_set.json'
    if not selected_fs_path.exists():
        logger.error("Selected feature set not found. Run 02_eval_fs_shap_mcda.py first.")
        return 1

    selected_fs = load_json(selected_fs_path)

    # Get top-N FS options to train on
    top_options = selected_fs.get('top_options', [selected_fs['selected_fs_option']])
    top_options_info = selected_fs.get('top_options_info', [])

    # If specific FS option requested, use only that one
    if args.fs_option:
        if args.fs_option in top_options:
            top_options = [args.fs_option]
            top_options_info = [info for info in top_options_info if info['fs_option'] == args.fs_option]
        else:
            logger.warning(f"Requested FS option '{args.fs_option}' not in top options.")
            fs_results_path = dirs['models'] / 'fs_results.json'
            if fs_results_path.exists():
                fs_results = load_json(fs_results_path)
                if args.fs_option in fs_results:
                    top_options = [args.fs_option]
                    top_options_info = [{
                        'fs_option': args.fs_option,
                        'selected_features': fs_results[args.fs_option]['selected_features'],
                        'n_features': len(fs_results[args.fs_option]['selected_features'])
                    }]

    # Build features dict for each FS option
    fs_features = {}
    for info in top_options_info:
        fs_features[info['fs_option']] = info['selected_features']

    # If we don't have features for some options, load from fs_results
    fs_results_path = dirs['models'] / 'fs_results.json'
    if fs_results_path.exists():
        fs_results = load_json(fs_results_path)
        for opt in top_options:
            if opt not in fs_features and opt in fs_results:
                fs_features[opt] = fs_results[opt]['selected_features']

    logger.info(f"Training models on {len(top_options)} FS options: {top_options}")

    # =========================================
    # Optimize Models for Each FS Option
    # =========================================
    all_fs_results = {}

    for fs_option in top_options:
        if fs_option not in fs_features:
            logger.warning(f"No features found for FS option '{fs_option}', skipping")
            continue

        logger.info("=" * 50)
        fs_result = optimize_for_fs_option(
            X_full, y, cv_plan, config,
            fs_option=fs_option,
            selected_features=fs_features[fs_option],
            output_dir=dirs['models'],
            logger=logger
        )
        all_fs_results[fs_option] = fs_result

    # =========================================
    # Generate Optimization Summary
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating optimization summaries...")

    # Summary per FS option
    summary_rows = []
    for fs_option, fs_result in all_fs_results.items():
        for model_name, results in fs_result['optimization_results'].items():
            summary_rows.append({
                'fs_option': fs_option,
                'n_features': fs_result['n_features'],
                'model': model_name,
                'status': 'failed' if 'error' in results else 'success',
                'best_fitness': results.get('best_fitness'),
                'best_params': str(results.get('best_params', {}))
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(dirs['tables'] / 'optimization_summary.csv', index=False)

    # FS comparison summary
    fs_comparison = []
    for fs_option, fs_result in all_fs_results.items():
        opt_results = fs_result['optimization_results']
        successful_models = [m for m, r in opt_results.items() if 'best_fitness' in r]
        avg_fitness = np.mean([opt_results[m]['best_fitness'] for m in successful_models]) if successful_models else np.inf
        best_fitness = min([opt_results[m]['best_fitness'] for m in successful_models]) if successful_models else np.inf
        best_model = min(successful_models, key=lambda m: opt_results[m]['best_fitness']) if successful_models else 'N/A'

        fs_comparison.append({
            'fs_option': fs_option,
            'n_features': fs_result['n_features'],
            'n_models_trained': len(successful_models),
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'best_model': best_model
        })

    fs_comparison_df = pd.DataFrame(fs_comparison).sort_values('best_fitness')
    fs_comparison_df.to_csv(dirs['tables'] / 'fs_comparison_summary.csv', index=False)

    # Save all FS results metadata
    save_json_numpy({
        'top_options': top_options,
        'fs_results': {
            k: {
                'fs_option': v['fs_option'],
                'n_features': v['n_features'],
                'features': v['features'],
                'trained_models': v['trained_models'],
                'output_dir': v['output_dir']
            }
            for k, v in all_fs_results.items()
        }
    }, dirs['tables'] / 'multi_fs_optimization_results.json')

    # =========================================
    # Generate Optimization Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating optimization plots...")

    for fs_option, fs_result in all_fs_results.items():
        for model_name, results in fs_result['optimization_results'].items():
            if 'history' in results and results['history']:
                plot_optimization_history(
                    results['history'],
                    title=f"Optimization - {model_name.upper()} ({fs_option})",
                    output_path=dirs['figures'] / f'optimization_{fs_option}_{model_name}.png'
                )

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Model Optimization Complete!")
    logger.info(f"  FS options trained: {len(all_fs_results)}")
    for fs_option, fs_result in all_fs_results.items():
        n_trained = len(fs_result['trained_models'])
        logger.info(f"    {fs_option}: {n_trained} models, {fs_result['n_features']} features")
    if not fs_comparison_df.empty:
        logger.info(f"  FS Comparison Summary:")
        for _, row in fs_comparison_df.iterrows():
            logger.info(f"    {row['fs_option']}: best_fitness={row['best_fitness']:.4f} ({row['best_model']})")
    logger.info(f"  Results saved to: {dirs['models']}")
    logger.info(f"  Tables saved to: {dirs['tables']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
