#!/usr/bin/env python
"""
Script 04: Evaluate Models and Apply Safeguards
================================================
Evaluate trained models with quarterly metrics and apply annual consistency safeguards.
Supports evaluation across multiple FS options for comparison.

Usage:
    python scripts/04_evaluate_and_safeguards.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/quarterly_metrics.csv (combined across all FS options)
    - outputs/runs/<run_id>/tables/annual_consistency.csv
    - outputs/runs/<run_id>/tables/fs_model_comparison.csv (comparison across FS options)
    - outputs/runs/<run_id>/predictions/<fs_option>/<model>_predictions.csv
    - outputs/runs/<run_id>/figures/predictions_<model>.png
    - outputs/runs/<run_id>/figures/fs_comparison_heatmap.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy, load_json, inverse_log_transform, get_latest_run_id
)
from src.data_io import load_processed_data
from src.splits import load_cv_plan, generate_cv_folds
from src.models import ModelRegistry
from src.evaluation import (
    compute_all_metrics, evaluate_by_horizon, evaluate_stability,
    create_evaluation_summary, compare_models
)
from src.safeguards import (
    check_annual_consistency, compare_with_annual_baseline,
    aggregate_quarterly_to_annual
)
from src.reporting import (
    plot_predictions_vs_actual, plot_horizon_comparison,
    plot_model_comparison, plot_annual_consistency, set_plot_style
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models and safeguards')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
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

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    set_plot_style()

    logger.info("=" * 60)
    logger.info("Script 04: Evaluate Models and Apply Safeguards (Multi-FS)")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    processed_dir = Path('data/processed')

    X_full = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    # Load multi-FS optimization results
    multi_fs_path = dirs['tables'] / 'multi_fs_optimization_results.json'
    selected_fs_path = dirs['tables'] / 'selected_feature_set.json'

    if multi_fs_path.exists():
        multi_fs_info = load_json(multi_fs_path)
        fs_options = multi_fs_info.get('top_options', [])
        fs_details = multi_fs_info.get('fs_results', {})
        logger.info(f"Found multi-FS training results for: {fs_options}")
    elif selected_fs_path.exists():
        # Fallback to single FS option (backward compatibility)
        selected_fs = load_json(selected_fs_path)
        fs_options = [selected_fs['selected_fs_option']]
        fs_details = {
            selected_fs['selected_fs_option']: {
                'features': selected_fs['selected_features'],
                'n_features': len(selected_fs['selected_features']),
                'output_dir': str(dirs['models'])
            }
        }
        logger.info(f"Using single FS option: {fs_options}")
    else:
        logger.error("No FS results found. Run previous scripts first.")
        return 1

    # =========================================
    # Evaluate Models for Each FS Option
    # =========================================
    all_fs_evaluations = {}
    combined_metrics = []

    for fs_option in fs_options:
        logger.info("=" * 50)
        logger.info(f"Evaluating models for FS option: {fs_option}")

        fs_info = fs_details.get(fs_option, {})
        selected_features = fs_info.get('features', [])
        fs_model_dir = Path(fs_info.get('output_dir', dirs['models'] / fs_option))

        if not selected_features:
            logger.warning(f"No features found for {fs_option}, skipping")
            continue

        # Prepare data for this FS option
        available_features = [f for f in selected_features if f in X_full.columns]
        X = X_full[available_features]

        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y_aligned = y.loc[common_idx]

        valid_mask = ~(X.isnull().any(axis=1) | y_aligned.isnull())
        X = X[valid_mask]
        y_aligned = y_aligned[valid_mask]

        logger.info(f"  Data: {len(X)} samples, {len(available_features)} features")

        # Load models for this FS option
        models = {}
        model_files = list(fs_model_dir.glob('*_model.pkl'))

        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                model = ModelRegistry.get(model_name).load(model_file)
                models[model_name] = model
                logger.info(f"    Loaded: {model_name}")
            except Exception as e:
                logger.warning(f"    Failed to load {model_name}: {e}")

        if not models:
            logger.warning(f"No trained models found for {fs_option}")
            continue

        # Create FS-specific predictions directory
        fs_pred_dir = dirs['predictions'] / fs_option
        fs_pred_dir.mkdir(parents=True, exist_ok=True)

        # =========================================
        # Walk-Forward Evaluation for this FS option
        # =========================================
        logger.info("-" * 40)
        logger.info(f"Running walk-forward evaluation for {fs_option}...")

        all_model_results = {}
        all_predictions = {}

        for model_name, model in models.items():
            logger.info(f"    Evaluating: {model_name}")

            predictions_list = []

            for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y_aligned, cv_plan):
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    for i, (date, actual, predicted) in enumerate(zip(
                        y_test.index, y_test.values, y_pred
                    )):
                        predictions_list.append({
                            'fold_id': fold.fold_id,
                            'horizon': fold.horizon,
                            'date': date,
                            'actual': actual,
                            'predicted': predicted
                        })
                except Exception as e:
                    logger.warning(f"      Fold {fold.fold_id} failed: {e}")

            if not predictions_list:
                continue

            predictions_df = pd.DataFrame(predictions_list)
            predictions_df.to_csv(fs_pred_dir / f'{model_name}_predictions.csv', index=False)
            all_predictions[model_name] = predictions_df

            # Compute metrics
            horizon_metrics = evaluate_by_horizon(predictions_df, config.splits.horizons)
            stability_metrics = evaluate_stability(predictions_df)

            summary = create_evaluation_summary(
                model_name, horizon_metrics, stability_metrics,
                config.splits.horizon_weights
            )
            all_model_results[model_name] = summary

            logger.info(f"      Weighted MAE: {summary['weighted_mae']:.4f}")

            # Add to combined metrics
            combined_metrics.append({
                'fs_option': fs_option,
                'n_features': len(available_features),
                'model': model_name,
                **summary
            })

        # Store FS-specific results
        all_fs_evaluations[fs_option] = {
            'model_results': all_model_results,
            'predictions': all_predictions,
            'features': available_features,
            'n_features': len(available_features)
        }

    # =========================================
    # Save Combined Metrics Across All FS Options
    # =========================================
    logger.info("-" * 40)
    logger.info("Saving combined metrics...")

    quarterly_df = pd.DataFrame(combined_metrics)
    quarterly_df.to_csv(dirs['tables'] / 'quarterly_metrics.csv', index=False)

    # Create FS comparison summary
    fs_model_comparison = quarterly_df.pivot_table(
        index='model',
        columns='fs_option',
        values='weighted_mae',
        aggfunc='first'
    )
    fs_model_comparison.to_csv(dirs['tables'] / 'fs_model_comparison.csv')
    logger.info(f"FS-Model comparison:\n{fs_model_comparison.to_string()}")

    # =========================================
    # Annual Consistency Safeguard (for all FS options)
    # =========================================
    logger.info("-" * 40)
    logger.info("Checking annual consistency...")

    annual_summary = []
    for fs_option, fs_eval in all_fs_evaluations.items():
        all_predictions = fs_eval['predictions']
        all_model_results = fs_eval['model_results']

        for model_name, predictions_df in all_predictions.items():
            consistency = check_annual_consistency(
                predictions_df,
                transform=config.data.target_transform
            )

            # Add to model results
            all_model_results[model_name]['annual_mae'] = consistency.get('annual_mae')
            all_model_results[model_name]['annual_mape'] = consistency.get('annual_mape')
            all_model_results[model_name]['annual_consistency'] = consistency.get('consistency_score')

            annual_summary.append({
                'fs_option': fs_option,
                'model': model_name,
                'annual_mae': consistency.get('annual_mae'),
                'annual_mape': consistency.get('annual_mape'),
                'annual_bias': consistency.get('annual_bias'),
                'consistency_score': consistency.get('consistency_score')
            })

            logger.info(f"  {fs_option}/{model_name}: Annual MAPE = {consistency.get('annual_mape', 0):.2f}%")

    pd.DataFrame(annual_summary).to_csv(dirs['tables'] / 'annual_consistency.csv', index=False)

    # =========================================
    # Annual Baseline Comparison
    # =========================================
    logger.info("-" * 40)
    logger.info("Comparing with annual baselines...")

    # Get annual actual values
    y_original = inverse_log_transform(y.values) if config.data.target_transform == 'log' else y.values
    y_annual = pd.Series(y_original, index=y.index).groupby(y.index.year).sum()

    baseline_comparisons = {}
    for fs_option, fs_eval in all_fs_evaluations.items():
        all_predictions = fs_eval['predictions']
        for model_name, predictions_df in all_predictions.items():
            comparison = compare_with_annual_baseline(
                predictions_df, y_annual,
                transform=config.data.target_transform
            )
            baseline_comparisons[f"{fs_option}_{model_name}"] = comparison

    save_json_numpy(baseline_comparisons, dirs['tables'] / 'annual_benchmark_comparison.json')

    # =========================================
    # Generate Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating plots...")

    # Model comparison plot (combined across all FS options)
    plot_model_comparison(
        quarterly_df,
        metrics=['weighted_mae', 'stability_score'],
        title="Model Comparison (All FS Options)",
        output_path=dirs['figures'] / 'model_comparison.png'
    )

    # FS comparison heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(fs_model_comparison, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax)
        ax.set_title('Model Performance Across FS Options (Weighted MAE)')
        ax.set_xlabel('FS Option')
        ax.set_ylabel('Model')
        plt.tight_layout()
        plt.savefig(dirs['figures'] / 'fs_comparison_heatmap.png', dpi=config.output.figure_dpi)
        plt.close()
        logger.info("Generated FS comparison heatmap")
    except Exception as e:
        logger.warning(f"Failed to generate heatmap: {e}")

    # =========================================
    # Predictions vs Actual for ALL Models (all FS options)
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating prediction plots for ALL models...")

    for fs_option, fs_eval in all_fs_evaluations.items():
        for model_name, predictions_df in fs_eval['predictions'].items():
            try:
                pred_agg = predictions_df.groupby('date').agg({
                    'actual': 'mean', 'predicted': 'mean'
                }).reset_index()

                plot_predictions_vs_actual(
                    pred_agg,
                    title=f"Predictions vs Actual ({fs_option}/{model_name})",
                    output_path=dirs['figures'] / f'predictions_{fs_option}_{model_name}.png'
                )
                logger.info(f"  Generated prediction plot: {fs_option}/{model_name}")
            except Exception as e:
                logger.warning(f"  Failed to plot predictions for {fs_option}/{model_name}: {e}")

    # =========================================
    # Coefficients/Feature Importance for ALL Models (per FS option)
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating coefficient/importance plots for ALL models...")

    from src.reporting.plots import plot_model_coefficients

    for fs_option, fs_eval in all_fs_evaluations.items():
        fs_info = fs_details.get(fs_option, {})
        fs_model_dir = Path(fs_info.get('output_dir', dirs['models'] / fs_option))
        available_features = fs_eval['features']

        # Load models for this FS option
        model_files = list(fs_model_dir.glob('*_model.pkl'))
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                model = ModelRegistry.get(model_name).load(model_file)

                # Get feature importance or coefficients
                if model_name.lower() == 'ridge':
                    # Ridge coefficients
                    if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                        coef_df = pd.DataFrame({
                            'feature': available_features,
                            'coefficient': model.model.coef_,
                            'abs_coefficient': np.abs(model.model.coef_)
                        }).sort_values('abs_coefficient', ascending=False)
                        coef_df.to_csv(dirs['tables'] / f'{fs_option}_{model_name}_coefficients.csv', index=False)
                        plot_model_coefficients(
                            coef_df, model_name,
                            output_path=dirs['figures'] / f'coefficients_{fs_option}_{model_name}.png'
                        )
                        logger.info(f"  Generated coefficients plot: {fs_option}/{model_name}")
                elif hasattr(model, 'get_feature_importance') and model.get_feature_importance():
                    # Tree-based importance
                    importance = model.get_feature_importance()
                    imp_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in importance.items()
                    ]).sort_values('importance', ascending=False)
                    imp_df.to_csv(dirs['tables'] / f'{fs_option}_{model_name}_feature_importance.csv', index=False)
                    plot_model_coefficients(
                        imp_df, model_name,
                        output_path=dirs['figures'] / f'coefficients_{fs_option}_{model_name}.png'
                    )
                    logger.info(f"  Generated importance plot: {fs_option}/{model_name}")
            except Exception as e:
                logger.warning(f"  Failed to plot coefficients for {fs_option}/{model_name}: {e}")

    # =========================================
    # SHAP Beeswarm Plots for ALL SHAP-Supporting Models (per FS option)
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating SHAP beeswarm plots for SHAP-supporting models...")

    from src.interpretability.shap_analysis import compute_shap_values
    from src.reporting.plots import plot_shap_beeswarm

    for fs_option, fs_eval in all_fs_evaluations.items():
        fs_info = fs_details.get(fs_option, {})
        fs_model_dir = Path(fs_info.get('output_dir', dirs['models'] / fs_option))
        available_features = fs_eval['features']

        # Prepare data for this FS option
        X_fs = X_full[[f for f in available_features if f in X_full.columns]]
        common_idx = X_fs.index.intersection(y.index)
        X_fs = X_fs.loc[common_idx]
        y_fs = y.loc[common_idx]
        valid_mask = ~(X_fs.isnull().any(axis=1) | y_fs.isnull())
        X_fs = X_fs[valid_mask]
        y_fs = y_fs[valid_mask]

        model_files = list(fs_model_dir.glob('*_model.pkl'))
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                model = ModelRegistry.get(model_name).load(model_file)

                if not (hasattr(model, 'supports_shap') and model.supports_shap):
                    logger.info(f"  {fs_option}/{model_name}: Does not support SHAP, skipping")
                    continue

                # Refit model on full data for SHAP
                model.fit(X_fs, y_fs)
                shap_values, _ = compute_shap_values(model.model, X_fs, 'tree')

                plot_shap_beeswarm(
                    shap_values, X_fs,
                    title=f"SHAP Beeswarm - {fs_option}/{model_name}",
                    output_path=dirs['figures'] / f'shap_beeswarm_{fs_option}_{model_name}.png'
                )
                logger.info(f"  Generated SHAP beeswarm: {fs_option}/{model_name}")
            except Exception as e:
                logger.warning(f"  SHAP beeswarm failed for {fs_option}/{model_name}: {e}")

    # =========================================
    # Horizon Comparison for Best Model per FS Option
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating horizon comparison plots...")

    for fs_option, fs_eval in all_fs_evaluations.items():
        all_model_results = fs_eval['model_results']
        if not all_model_results:
            continue

        # Find best model for this FS option
        best_model = min(all_model_results.items(),
                        key=lambda x: x[1].get('weighted_mae', float('inf')))[0]

        try:
            horizon_metrics = {
                h: all_model_results[best_model].get(f'mae_h{h}', 0)
                for h in config.splits.horizons
            }
            plot_horizon_comparison(
                {h: {'mae': v} for h, v in horizon_metrics.items()},
                title=f"MAE by Horizon ({fs_option}/{best_model})",
                output_path=dirs['figures'] / f'horizon_comparison_{fs_option}.png'
            )
            logger.info(f"  Generated horizon comparison: {fs_option}")
        except Exception as e:
            logger.warning(f"  Failed to generate horizon comparison for {fs_option}: {e}")

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Evaluation and Safeguards Complete!")
    logger.info(f"  FS options evaluated: {len(all_fs_evaluations)}")

    # Summary per FS option
    for fs_option, fs_eval in all_fs_evaluations.items():
        model_results = fs_eval['model_results']
        n_models = len(model_results)
        if model_results:
            best_mae = min(r.get('weighted_mae', float('inf')) for r in model_results.values())
            best_model = min(model_results.items(),
                            key=lambda x: x[1].get('weighted_mae', float('inf')))[0]
            logger.info(f"  {fs_option}: {n_models} models, best={best_model} (MAE={best_mae:.4f})")

    # Overall best
    if combined_metrics:
        overall_best = min(combined_metrics, key=lambda x: x.get('weighted_mae', float('inf')))
        logger.info(f"  Overall best: {overall_best['fs_option']}/{overall_best['model']} "
                   f"(MAE={overall_best['weighted_mae']:.4f})")

    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Predictions saved to: {dirs['predictions']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
