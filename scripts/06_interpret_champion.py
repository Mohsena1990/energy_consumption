#!/usr/bin/env python
"""
Script 06: Interpret Champion Model
===================================
Generate SHAP explanations and regime analysis for the champion model.

Usage:
    python scripts/06_interpret_champion.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/champion_interpretation.json
    - outputs/runs/<run_id>/tables/top_drivers.csv
    - outputs/runs/<run_id>/figures/shap_summary.png
    - outputs/runs/<run_id>/figures/shap_regime_compare.png
    - outputs/runs/<run_id>/figures/seasonal_leverage.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy, load_json, get_latest_run_id
)
from src.data_io import load_processed_data
from src.models import ModelRegistry
from src.interpretability import (
    compute_shap_values, get_feature_importance_from_shap,
    analyze_regime_shap, analyze_seasonal_shap,
    compute_permutation_importance, get_ridge_coefficients,
    generate_interpretation_report
)
from src.reporting import (
    plot_feature_importance, plot_shap_summary, plot_regime_comparison,
    set_plot_style
)


def parse_args():
    parser = argparse.ArgumentParser(description='Interpret champion model')
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
    logger.info("Script 06: Interpret Champion Model")
    logger.info("=" * 60)

    # Load champion pipeline info
    champion_path = dirs['tables'] / 'champion_pipeline.json'
    if not champion_path.exists():
        logger.error("Champion pipeline not found. Run 05_select_best_model.py first.")
        return 1

    champion_info = load_json(champion_path)
    champion_model_name = champion_info['champion_model']

    logger.info(f"Champion model: {champion_model_name}")

    # Load data
    logger.info("Loading data...")
    processed_dir = Path('data/processed')

    X_full = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']

    # Get selected features
    selected_features = champion_info.get('selected_features', list(X_full.columns))
    available_features = [f for f in selected_features if f in X_full.columns]
    X = X_full[available_features]

    # Align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Data: {len(X)} samples, {len(X.columns)} features")

    # Load champion model
    model_path = dirs['models'] / f'{champion_model_name}_model.pkl'
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1

    model = ModelRegistry.get(champion_model_name).load(model_path)
    logger.info(f"Loaded model: {champion_model_name}")

    # =========================================
    # Feature Importance Analysis
    # =========================================
    logger.info("-" * 40)
    logger.info("Computing feature importance...")

    interpretation_results = {
        'model': champion_model_name,
        'n_features': len(X.columns),
        'n_samples': len(X)
    }

    if champion_model_name == 'ridge':
        # Use coefficients for Ridge
        logger.info("Using Ridge coefficients...")
        coef_df = get_ridge_coefficients(model)
        interpretation_results['method'] = 'coefficients'
        interpretation_results['coefficients'] = coef_df.to_dict(orient='records')

        # Also compute permutation importance
        perm_df = compute_permutation_importance(model, X, y)
        interpretation_results['permutation_importance'] = perm_df.to_dict(orient='records')

        # Plot
        plot_feature_importance(
            coef_df.rename(columns={'abs_coefficient': 'importance'}),
            title=f"Ridge Coefficients (Absolute)",
            output_path=dirs['figures'] / 'ridge_coefficients.png'
        )

    elif hasattr(model, 'supports_shap') and model.supports_shap:
        # Use SHAP for tree-based models
        logger.info("Computing SHAP values...")

        try:
            import shap

            shap_values, explainer = compute_shap_values(model, X, 'tree')
            interpretation_results['method'] = 'shap'

            # Feature importance from SHAP
            importance_df = get_feature_importance_from_shap(shap_values, list(X.columns))
            interpretation_results['shap_importance'] = importance_df.to_dict(orient='records')

            # Save top drivers
            importance_df.to_csv(dirs['tables'] / 'top_drivers.csv', index=False)

            # SHAP summary plot
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, max_display=15, show=False)
            plt.title(f"SHAP Summary - {champion_model_name}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(dirs['figures'] / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Feature importance bar plot
            plot_feature_importance(
                importance_df,
                title=f"Feature Importance (SHAP) - {champion_model_name}",
                output_path=dirs['figures'] / 'feature_importance.png'
            )

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            logger.info("Falling back to permutation importance...")

            perm_df = compute_permutation_importance(model, X, y)
            interpretation_results['method'] = 'permutation'
            interpretation_results['permutation_importance'] = perm_df.to_dict(orient='records')

            plot_feature_importance(
                perm_df.rename(columns={'importance_mean': 'importance'}),
                title=f"Permutation Importance - {champion_model_name}",
                output_path=dirs['figures'] / 'feature_importance.png'
            )

    else:
        # Fallback to permutation importance
        logger.info("Using permutation importance...")
        perm_df = compute_permutation_importance(model, X, y)
        interpretation_results['method'] = 'permutation'
        interpretation_results['permutation_importance'] = perm_df.to_dict(orient='records')

        plot_feature_importance(
            perm_df.rename(columns={'importance_mean': 'importance'}),
            title=f"Permutation Importance - {champion_model_name}",
            output_path=dirs['figures'] / 'feature_importance.png'
        )

    # =========================================
    # Regime Analysis (if SHAP available)
    # =========================================
    if hasattr(model, 'supports_shap') and model.supports_shap:
        logger.info("-" * 40)
        logger.info("Analyzing feature importance by regime...")

        try:
            regime_results = analyze_regime_shap(
                model, X,
                regime_periods={
                    'pre_covid': (None, '2020-01-01'),
                    'covid': ('2020-01-01', '2022-01-01'),
                    'post_covid': ('2022-01-01', None)
                },
                model_type='tree'
            )

            interpretation_results['regime_analysis'] = {
                k: v.to_dict(orient='records') for k, v in regime_results.items()
            }

            # Plot regime comparison
            if regime_results:
                plot_regime_comparison(
                    regime_results,
                    title="Feature Importance by Regime",
                    output_path=dirs['figures'] / 'shap_regime_compare.png'
                )

        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")

    # =========================================
    # Seasonal Analysis
    # =========================================
    logger.info("-" * 40)
    logger.info("Analyzing seasonal patterns...")

    try:
        seasonal_df = analyze_seasonal_shap(model, X, model_type='tree')

        if len(seasonal_df) > 0:
            interpretation_results['seasonal_analysis'] = seasonal_df.to_dict(orient='records')
            seasonal_df.to_csv(dirs['tables'] / 'seasonal_importance.csv', index=False)

            # Create seasonal leverage plot
            fig, ax = plt.subplots(figsize=(12, 6))

            pivot = seasonal_df.pivot(index='feature', columns='quarter', values='importance')
            top_features = seasonal_df.groupby('feature')['importance'].mean().nlargest(10).index

            pivot.loc[top_features].plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            ax.set_title('Seasonal Feature Leverage (Top 10 Features)', fontweight='bold')
            ax.legend(title='Quarter')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(dirs['figures'] / 'seasonal_leverage.png', dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.warning(f"Seasonal analysis failed: {e}")

    # =========================================
    # Save Interpretation Results
    # =========================================
    save_json_numpy(interpretation_results, dirs['tables'] / 'champion_interpretation.json')

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Champion Model Interpretation Complete!")
    logger.info(f"  Model: {champion_model_name}")
    logger.info(f"  Method: {interpretation_results.get('method', 'unknown')}")

    if 'shap_importance' in interpretation_results:
        top_feature = interpretation_results['shap_importance'][0]['feature']
        logger.info(f"  Top feature: {top_feature}")
    elif 'coefficients' in interpretation_results:
        top_feature = interpretation_results['coefficients'][0]['feature']
        logger.info(f"  Top feature: {top_feature}")

    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
