#!/usr/bin/env python
"""
Script 05: Select Best Model with Pareto + MCDA
===============================================
Apply Pareto filtering and MCDA to select the champion model.

Usage:
    python scripts/05_select_best_model.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/pareto_front.csv
    - outputs/runs/<run_id>/tables/mcda_model_ranking.csv
    - outputs/runs/<run_id>/tables/champion_pipeline.json
    - outputs/runs/<run_id>/figures/pareto_front.png
    - outputs/runs/<run_id>/figures/mcda_model_ranking.png
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
from src.decision import pareto_filter, vikor, topsis, select_best_model
from src.reporting import (
    plot_pareto_front, plot_mcda_ranking, set_plot_style
)


def parse_args():
    parser = argparse.ArgumentParser(description='Select best model with MCDA')
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
    logger.info("Script 05: Select Best Model with Pareto + MCDA")
    logger.info("=" * 60)

    # Load evaluation results
    logger.info("Loading evaluation results...")

    quarterly_path = dirs['tables'] / 'quarterly_metrics.csv'
    annual_path = dirs['tables'] / 'annual_consistency.csv'

    if not quarterly_path.exists():
        logger.error("Quarterly metrics not found. Run 04_evaluate_and_safeguards.py first.")
        return 1

    quarterly_df = pd.read_csv(quarterly_path)
    annual_df = pd.read_csv(annual_path) if annual_path.exists() else None

    # Merge metrics
    if annual_df is not None:
        metrics_df = quarterly_df.merge(annual_df, on='model', how='left')
    else:
        metrics_df = quarterly_df

    logger.info(f"Loaded metrics for {len(metrics_df)} models")
    logger.info(f"Available metrics: {list(metrics_df.columns)}")

    # =========================================
    # Prepare Criteria for MCDA
    # =========================================
    logger.info("-" * 40)
    logger.info("Preparing criteria for MCDA...")

    # Define criteria (adjust based on available columns)
    criteria = []
    criteria_types = {}

    if 'weighted_mae' in metrics_df.columns:
        criteria.append('weighted_mae')
        criteria_types['weighted_mae'] = 'cost'

    if 'stability_score' in metrics_df.columns:
        criteria.append('stability_score')
        criteria_types['stability_score'] = 'benefit'
    elif 'error_std' in metrics_df.columns:
        criteria.append('error_std')
        criteria_types['error_std'] = 'cost'

    if 'annual_consistency' in metrics_df.columns:
        criteria.append('annual_consistency')
        criteria_types['annual_consistency'] = 'benefit'
    elif 'annual_mape' in metrics_df.columns:
        criteria.append('annual_mape')
        criteria_types['annual_mape'] = 'cost'

    logger.info(f"Criteria: {criteria}")

    if len(criteria) < 2:
        logger.warning("Not enough criteria, using available numeric columns")
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        criteria = [c for c in numeric_cols if c != 'model'][:3]
        criteria_types = {c: 'cost' if 'mae' in c.lower() or 'error' in c.lower() else 'benefit'
                        for c in criteria}

    # =========================================
    # Pareto Filtering
    # =========================================
    logger.info("-" * 40)
    logger.info("Applying Pareto filter...")

    pareto_df = pareto_filter(metrics_df.copy(), criteria, criteria_types)
    pareto_df.to_csv(dirs['tables'] / 'pareto_front.csv', index=False)

    logger.info(f"Pareto front: {len(pareto_df)} models (from {len(metrics_df)})")

    # =========================================
    # MCDA Ranking
    # =========================================
    logger.info("-" * 40)
    logger.info(f"Applying {config.mcda.method.upper()} ranking...")

    # Prepare weights
    weights = {}
    for c in criteria:
        if 'mae' in c.lower():
            weights[c] = config.mcda.model_weights.get('quarterly_mae', 0.35)
        elif 'stability' in c.lower():
            weights[c] = config.mcda.model_weights.get('stability', 0.2)
        elif 'annual' in c.lower():
            weights[c] = config.mcda.model_weights.get('annual_consistency', 0.25)
        else:
            weights[c] = 1.0 / len(criteria)

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    logger.info(f"Weights: {weights}")

    # Apply MCDA
    if config.mcda.method == 'vikor':
        ranking_df = vikor(
            pareto_df.copy(), criteria, weights, criteria_types,
            v=config.mcda.vikor_v
        )
        score_col = 'vikor_Q'
        rank_col = 'vikor_rank'
    else:
        ranking_df = topsis(
            pareto_df.copy(), criteria, weights, criteria_types
        )
        score_col = 'topsis_score'
        rank_col = 'topsis_rank'

    ranking_df.to_csv(dirs['tables'] / 'mcda_model_ranking.csv', index=False)

    # Get champion model
    champion_model = ranking_df.iloc[0]['model']
    champion_score = ranking_df.iloc[0][score_col]

    logger.info(f"Champion model: {champion_model} (score: {champion_score:.4f})")

    # =========================================
    # Create Champion Pipeline Record
    # =========================================
    logger.info("-" * 40)
    logger.info("Creating champion pipeline record...")

    champion_metrics = ranking_df.iloc[0].to_dict()

    # Load best params
    best_params_path = dirs['tables'] / 'best_params_all_models.json'
    best_params = {}
    if best_params_path.exists():
        all_params = load_json(best_params_path)
        best_params = all_params.get(champion_model, {}).get('params', {})

    # Load selected features
    selected_fs_path = dirs['tables'] / 'selected_feature_set.json'
    selected_features = []
    if selected_fs_path.exists():
        fs_info = load_json(selected_fs_path)
        selected_features = fs_info.get('selected_features', [])

    champion_pipeline = {
        'champion_model': champion_model,
        'mcda_method': config.mcda.method,
        'mcda_score': champion_score,
        'best_params': best_params,
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'metrics': {
            k: v for k, v in champion_metrics.items()
            if not k.startswith('vikor_') and not k.startswith('topsis_') and k != 'model'
        },
        'ranking': ranking_df[['model', score_col, rank_col]].to_dict(orient='records'),
        'criteria': criteria,
        'weights': weights
    }

    save_json_numpy(champion_pipeline, dirs['tables'] / 'champion_pipeline.json')

    # =========================================
    # Generate Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating plots...")

    # Pareto front plot
    if len(criteria) >= 2:
        plot_pareto_front(
            pareto_df,
            obj1=criteria[0],
            obj2=criteria[1],
            all_points_df=metrics_df,
            title="Model Selection: Pareto Front",
            output_path=dirs['figures'] / 'pareto_front.png'
        )

    # MCDA ranking plot
    plot_mcda_ranking(
        ranking_df,
        score_col=score_col,
        title=f"Model Ranking ({config.mcda.method.upper()})",
        output_path=dirs['figures'] / 'mcda_model_ranking.png'
    )

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Model Selection Complete!")
    logger.info(f"  Champion model: {champion_model}")
    logger.info(f"  MCDA method: {config.mcda.method}")
    logger.info(f"  MCDA score: {champion_score:.4f}")
    logger.info(f"  Pareto front size: {len(pareto_df)}")
    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
