#!/usr/bin/env python
"""
Script 04: Evaluate Models and Apply Safeguards
================================================
Evaluate trained models with quarterly metrics and apply annual consistency safeguards.

Usage:
    python scripts/04_evaluate_and_safeguards.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/quarterly_metrics.csv
    - outputs/runs/<run_id>/tables/annual_consistency.csv
    - outputs/runs/<run_id>/tables/annual_benchmark_comparison.csv
    - outputs/runs/<run_id>/predictions/<model>_predictions.csv
    - outputs/runs/<run_id>/figures/predictions_<model>.png
    - outputs/runs/<run_id>/figures/annual_consistency.png
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
    logger.info("Script 04: Evaluate Models and Apply Safeguards")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    processed_dir = Path('data/processed')

    X_full = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    # Load selected features
    selected_fs = load_json(dirs['tables'] / 'selected_feature_set.json')
    selected_features = selected_fs['selected_features']

    available_features = [f for f in selected_features if f in X_full.columns]
    X = X_full[available_features]

    # Align and clean
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Evaluation data: {len(X)} samples")

    # =========================================
    # Load Trained Models
    # =========================================
    logger.info("-" * 40)
    logger.info("Loading trained models...")

    models = {}
    model_files = list(dirs['models'].glob('*_model.pkl'))

    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '')
        try:
            model = ModelRegistry.get(model_name).load(model_file)
            models[model_name] = model
            logger.info(f"  Loaded: {model_name}")
        except Exception as e:
            logger.warning(f"  Failed to load {model_name}: {e}")

    if not models:
        logger.error("No trained models found. Run 03_optimize_models.py first.")
        return 1

    # =========================================
    # Walk-Forward Evaluation
    # =========================================
    logger.info("-" * 40)
    logger.info("Running walk-forward evaluation...")

    all_model_results = {}
    all_predictions = {}

    for model_name, model in models.items():
        logger.info(f"  Evaluating: {model_name}")

        predictions_list = []

        for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
            try:
                # Re-fit model on training fold
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
                logger.warning(f"    Fold {fold.fold_id} failed: {e}")

        if not predictions_list:
            continue

        predictions_df = pd.DataFrame(predictions_list)
        predictions_df.to_csv(dirs['predictions'] / f'{model_name}_predictions.csv', index=False)
        all_predictions[model_name] = predictions_df

        # Compute metrics
        horizon_metrics = evaluate_by_horizon(predictions_df, config.splits.horizons)
        stability_metrics = evaluate_stability(predictions_df)

        summary = create_evaluation_summary(
            model_name, horizon_metrics, stability_metrics,
            config.splits.horizon_weights
        )
        all_model_results[model_name] = summary

        logger.info(f"    Weighted MAE: {summary['weighted_mae']:.4f}")

    # Save quarterly metrics
    quarterly_df = compare_models(all_model_results)
    quarterly_df.to_csv(dirs['tables'] / 'quarterly_metrics.csv', index=False)

    # =========================================
    # Annual Consistency Safeguard
    # =========================================
    logger.info("-" * 40)
    logger.info("Checking annual consistency...")

    annual_results = {}

    for model_name, predictions_df in all_predictions.items():
        consistency = check_annual_consistency(
            predictions_df,
            transform=config.data.target_transform
        )
        annual_results[model_name] = consistency

        # Add to model results
        all_model_results[model_name]['annual_mae'] = consistency.get('annual_mae')
        all_model_results[model_name]['annual_mape'] = consistency.get('annual_mape')
        all_model_results[model_name]['annual_consistency'] = consistency.get('consistency_score')

        logger.info(f"  {model_name}: Annual MAPE = {consistency.get('annual_mape', 0):.2f}%")

    # Save annual consistency results
    annual_summary = []
    for model_name, results in annual_results.items():
        annual_summary.append({
            'model': model_name,
            'annual_mae': results.get('annual_mae'),
            'annual_mape': results.get('annual_mape'),
            'annual_bias': results.get('annual_bias'),
            'consistency_score': results.get('consistency_score')
        })
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
    for model_name, predictions_df in all_predictions.items():
        comparison = compare_with_annual_baseline(
            predictions_df, y_annual,
            transform=config.data.target_transform
        )
        baseline_comparisons[model_name] = comparison

    save_json_numpy(baseline_comparisons, dirs['tables'] / 'annual_benchmark_comparison.json')

    # =========================================
    # Generate Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating plots...")

    # Model comparison
    updated_quarterly_df = compare_models(all_model_results)
    updated_quarterly_df.to_csv(dirs['tables'] / 'quarterly_metrics.csv', index=False)

    plot_model_comparison(
        updated_quarterly_df,
        metrics=['weighted_mae', 'stability_score', 'annual_consistency'],
        title="Model Comparison",
        output_path=dirs['figures'] / 'model_comparison.png'
    )

    # =========================================
    # Predictions vs Actual for ALL Models
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating prediction plots for ALL models...")

    for model_name, predictions_df in all_predictions.items():
        try:
            pred_agg = predictions_df.groupby('date').agg({
                'actual': 'mean', 'predicted': 'mean'
            }).reset_index()

            plot_predictions_vs_actual(
                pred_agg,
                title=f"Predictions vs Actual ({model_name})",
                output_path=dirs['figures'] / f'predictions_{model_name}.png'
            )
            logger.info(f"  Generated prediction plot: {model_name}")
        except Exception as e:
            logger.warning(f"  Failed to plot predictions for {model_name}: {e}")

    # =========================================
    # Coefficients/Feature Importance for ALL Models
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating coefficient/importance plots for ALL models...")

    from src.reporting.plots import plot_model_coefficients

    for model_name, model in models.items():
        try:
            # Get feature importance or coefficients
            if model_name.lower() == 'ridge':
                # Ridge coefficients
                if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'feature': available_features,
                        'coefficient': model.model.coef_,
                        'abs_coefficient': np.abs(model.model.coef_)
                    }).sort_values('abs_coefficient', ascending=False)
                    coef_df.to_csv(dirs['tables'] / f'{model_name}_coefficients.csv', index=False)
                    plot_model_coefficients(
                        coef_df, model_name,
                        output_path=dirs['figures'] / f'coefficients_{model_name}.png'
                    )
                    logger.info(f"  Generated coefficients plot: {model_name}")
            elif hasattr(model, 'get_feature_importance') and model.get_feature_importance():
                # Tree-based importance
                importance = model.get_feature_importance()
                imp_df = pd.DataFrame([
                    {'feature': k, 'importance': v}
                    for k, v in importance.items()
                ]).sort_values('importance', ascending=False)
                imp_df.to_csv(dirs['tables'] / f'{model_name}_feature_importance.csv', index=False)
                plot_model_coefficients(
                    imp_df, model_name,
                    output_path=dirs['figures'] / f'coefficients_{model_name}.png'
                )
                logger.info(f"  Generated importance plot: {model_name}")
        except Exception as e:
            logger.warning(f"  Failed to plot coefficients for {model_name}: {e}")

    # =========================================
    # SHAP Beeswarm Plots for ALL SHAP-Supporting Models
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating SHAP beeswarm plots for SHAP-supporting models...")

    from src.interpretability.shap_analysis import compute_shap_values
    from src.reporting.plots import plot_shap_beeswarm

    for model_name, model in models.items():
        if not (hasattr(model, 'supports_shap') and model.supports_shap):
            logger.info(f"  {model_name}: Does not support SHAP, skipping")
            continue

        try:
            # Refit model on full data for SHAP
            model.fit(X, y)
            shap_values, _ = compute_shap_values(model.model, X, 'tree')

            plot_shap_beeswarm(
                shap_values, X,
                title=f"SHAP Beeswarm - {model_name}",
                output_path=dirs['figures'] / f'shap_beeswarm_{model_name}.png'
            )
            logger.info(f"  Generated SHAP beeswarm: {model_name}")
        except Exception as e:
            logger.warning(f"  SHAP beeswarm failed for {model_name}: {e}")

    # Annual consistency plot (best model)
    if all_model_results:
        best_model = min(all_model_results.items(), key=lambda x: x[1].get('weighted_mae', float('inf')))[0]

    if annual_results:
        best_annual = annual_results.get(best_model, {})
        if 'annual_data' in best_annual:
            annual_df = pd.DataFrame(best_annual['annual_data'])
            plot_annual_consistency(
                annual_df,
                title=f"Annual Consistency ({best_model})",
                output_path=dirs['figures'] / 'annual_consistency.png'
            )

    # Horizon comparison
    if all_model_results:
        horizon_metrics = {
            h: all_model_results[best_model].get(f'mae_h{h}', 0)
            for h in config.splits.horizons
        }
        plot_horizon_comparison(
            {h: {'mae': v} for h, v in horizon_metrics.items()},
            title=f"MAE by Horizon ({best_model})",
            output_path=dirs['figures'] / 'horizon_comparison.png'
        )

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Evaluation and Safeguards Complete!")
    logger.info(f"  Models evaluated: {len(models)}")
    logger.info(f"  Best quarterly MAE: {min(r.get('weighted_mae', float('inf')) for r in all_model_results.values()):.4f}")
    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Predictions saved to: {dirs['predictions']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
