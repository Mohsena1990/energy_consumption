#!/usr/bin/env python
"""
Script 02: Evaluate Feature Selection with SHAP + MCDA
======================================================
Evaluate FS options using SHAP-based metrics and select the best using MCDA.

Usage:
    python scripts/02_eval_fs_shap_mcda.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/fs_evaluation_matrix.csv
    - outputs/runs/<run_id>/tables/fs_mcda_ranking.csv
    - outputs/runs/<run_id>/tables/selected_feature_set.json
    - outputs/runs/<run_id>/figures/shap_fs_*.png
    - outputs/runs/<run_id>/figures/fs_mcda_ranking.png
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
from src.fs import create_fs_evaluation_matrix
from src.decision import select_best_fs_option, pareto_filter
from src.reporting import (
    plot_mcda_ranking, plot_pareto_front, plot_feature_importance,
    set_plot_style
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FS with SHAP + MCDA')
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
        # Reuse the latest run directory so outputs from previous scripts are found
        latest = get_latest_run_id(config.output.base_dir)
        if latest:
            config.run_id = latest

    set_seed(config.seed)
    dirs = create_run_directories(config)

    logger = setup_logging(log_dir=dirs['logs'], run_id=config.run_id)
    set_plot_style()

    logger.info("=" * 60)
    logger.info("Script 02: Evaluate FS with SHAP + MCDA")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data and FS results...")
    processed_dir = Path('data/processed')

    X = load_processed_data(processed_dir / 'X_full')
    y = load_processed_data(processed_dir / 'y')['target']
    cv_plan = load_cv_plan(processed_dir / 'cv_plan.pkl')

    # Load FS results
    fs_results_path = dirs['models'] / 'fs_results.json'
    if not fs_results_path.exists():
        logger.error("FS results not found. Run 01_run_fs.py first.")
        return 1

    fs_results = load_json(fs_results_path)

    # =========================================
    # Evaluate FS Options with SHAP
    # =========================================
    logger.info("-" * 40)
    logger.info("Evaluating FS options with SHAP...")

    eval_matrix, detailed_results = create_fs_evaluation_matrix(
        fs_results, X, y, cv_plan, config
    )

    # Save evaluation matrix
    eval_matrix.to_csv(dirs['tables'] / 'fs_evaluation_matrix.csv', index=False)
    save_json_numpy(detailed_results, dirs['tables'] / 'fs_evaluation_detailed.json')

    logger.info(f"Evaluation matrix:\n{eval_matrix.to_string()}")

    # =========================================
    # MCDA Selection of Best FS Option
    # =========================================
    logger.info("-" * 40)
    logger.info("Selecting best FS option with MCDA...")

    best_fs_option, ranking_df, selection_details = select_best_fs_option(
        eval_matrix, config, use_pareto=config.mcda.use_pareto_filter
    )

    # Save MCDA results
    ranking_df.to_csv(dirs['tables'] / 'fs_mcda_ranking.csv', index=False)
    save_json_numpy(selection_details, dirs['tables'] / 'fs_selection_details.json')

    # Get selected features
    selected_features = fs_results[best_fs_option]['selected_features']

    selected_feature_set = {
        'selected_fs_option': best_fs_option,
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'mcda_method': config.mcda.method,
        'mcda_score': selection_details['best_score']
    }
    save_json_numpy(selected_feature_set, dirs['tables'] / 'selected_feature_set.json')

    logger.info(f"Selected FS option: {best_fs_option}")
    logger.info(f"Selected features ({len(selected_features)}): {selected_features[:10]}...")

    # =========================================
    # Generate Plots
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating plots...")

    # MCDA ranking plot
    score_col = 'vikor_Q' if config.mcda.method == 'vikor' else 'topsis_score'
    plot_mcda_ranking(
        ranking_df,
        score_col=score_col,
        title=f"FS Option Ranking ({config.mcda.method.upper()})",
        output_path=dirs['figures'] / 'fs_mcda_ranking.png'
    )

    # Pareto front plot
    criteria = ['C1_accuracy', 'C2_stability']
    criteria_types = {'C1_accuracy': 'cost', 'C2_stability': 'benefit'}

    if 'fs_option' in eval_matrix.columns:
        eval_matrix_indexed = eval_matrix.set_index('fs_option')
    else:
        eval_matrix_indexed = eval_matrix

    pareto_df = pareto_filter(eval_matrix.copy(), criteria, criteria_types)

    plot_pareto_front(
        pareto_df,
        obj1='C1_accuracy',
        obj2='C2_stability',
        all_points_df=eval_matrix,
        title="FS Options: Accuracy vs Stability (Pareto Front)",
        output_path=dirs['figures'] / 'fs_pareto_front.png'
    )

    # Feature importance for selected FS
    if best_fs_option in detailed_results:
        shap_importance = detailed_results[best_fs_option].get('avg_shap_importance', {})
        if shap_importance:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in shap_importance.items()
            ])
            plot_feature_importance(
                importance_df,
                title=f"Feature Importance ({best_fs_option})",
                output_path=dirs['figures'] / f'shap_importance_{best_fs_option}.png'
            )

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("FS Evaluation with SHAP + MCDA Complete!")
    logger.info(f"  Best FS option: {best_fs_option}")
    logger.info(f"  Number of features: {len(selected_features)}")
    logger.info(f"  MCDA method: {config.mcda.method}")
    logger.info(f"  MCDA score: {selection_details['best_score']:.4f}")
    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
