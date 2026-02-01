#!/usr/bin/env python
"""
Script 05: Select Best Model with Pareto + MCDA (Multi-FS Support)
===================================================================
Apply Pareto filtering and MCDA to select the champion model across all FS options.
Compares models trained on different feature sets and selects the best FS-model combination.

Usage:
    python scripts/05_select_best_model.py [--config CONFIG_PATH] [--run-id RUN_ID]

Outputs:
    - outputs/runs/<run_id>/tables/pareto_front.csv
    - outputs/runs/<run_id>/tables/mcda_model_ranking.csv
    - outputs/runs/<run_id>/tables/champion_pipeline.json
    - outputs/runs/<run_id>/tables/fs_model_full_comparison.csv
    - outputs/runs/<run_id>/figures/pareto_front.png
    - outputs/runs/<run_id>/figures/mcda_model_ranking.png
    - outputs/runs/<run_id>/figures/fs_model_comparison_3d.png
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
    logger.info("Script 05: Select Best Model with Pareto + MCDA (Multi-FS)")
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

    # Check if multi-FS format (has fs_option column)
    is_multi_fs = 'fs_option' in quarterly_df.columns
    if is_multi_fs:
        logger.info(f"Multi-FS mode: Found {quarterly_df['fs_option'].nunique()} FS options")
        fs_options = quarterly_df['fs_option'].unique().tolist()
        logger.info(f"  FS options: {fs_options}")

        # Create unique model identifier combining fs_option and model name
        quarterly_df['model_id'] = quarterly_df['fs_option'] + '/' + quarterly_df['model']
    else:
        logger.info("Single-FS mode")
        quarterly_df['model_id'] = quarterly_df['model']

    # Merge with annual metrics
    if annual_df is not None:
        if is_multi_fs and 'fs_option' in annual_df.columns:
            # Multi-FS: merge on both fs_option and model
            metrics_df = quarterly_df.merge(
                annual_df, on=['fs_option', 'model'], how='left', suffixes=('', '_annual')
            )
        else:
            # Single-FS: merge on model only
            metrics_df = quarterly_df.merge(annual_df, on='model', how='left', suffixes=('', '_annual'))
    else:
        metrics_df = quarterly_df

    logger.info(f"Loaded metrics for {len(metrics_df)} model configurations")
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

    logger.info(f"Pareto front: {len(pareto_df)} model configs (from {len(metrics_df)})")
    if is_multi_fs:
        pareto_fs_counts = pareto_df['fs_option'].value_counts()
        logger.info(f"  Pareto front by FS: {pareto_fs_counts.to_dict()}")

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

    # Get top-N configurations from ranking
    top_n = config.mcda.top_n_options
    top_configs = ranking_df.head(top_n)

    if is_multi_fs:
        # Multi-FS: Use model_id as the identifier
        top_model_ids = top_configs['model_id'].tolist()
        champion_row = ranking_df.iloc[0]
        champion_model_id = champion_row['model_id']
        champion_fs = champion_row['fs_option']
        champion_model = champion_row['model']
        champion_score = champion_row[score_col]

        logger.info(f"Top {top_n} configurations: {top_model_ids}")
        logger.info(f"Champion: {champion_model_id} (score: {champion_score:.4f})")
        logger.info(f"  FS option: {champion_fs}")
        logger.info(f"  Model: {champion_model}")
    else:
        # Single-FS mode
        top_model_ids = top_configs['model'].tolist()
        champion_row = ranking_df.iloc[0]
        champion_model_id = champion_row['model']
        champion_model = champion_model_id
        champion_fs = None
        champion_score = champion_row[score_col]

        logger.info(f"Top {top_n} models: {top_model_ids}")
        logger.info(f"Champion model: {champion_model} (score: {champion_score:.4f})")

    # =========================================
    # Create Champion Pipeline Record
    # =========================================
    logger.info("-" * 40)
    logger.info("Creating champion pipeline record...")

    champion_metrics = ranking_df.iloc[0].to_dict()

    # Load best params (from FS-specific directory if multi-FS)
    best_params = {}
    if is_multi_fs and champion_fs:
        fs_params_path = dirs['models'] / champion_fs / 'best_params_all_models.json'
        if fs_params_path.exists():
            all_params = load_json(fs_params_path)
            best_params = all_params.get(champion_model, {}).get('params', {})
    else:
        best_params_path = dirs['tables'] / 'best_params_all_models.json'
        if best_params_path.exists():
            all_params = load_json(best_params_path)
            best_params = all_params.get(champion_model, {}).get('params', {})

    # Load selected features (from multi-FS results or selected_feature_set)
    selected_features = []
    multi_fs_path = dirs['tables'] / 'multi_fs_optimization_results.json'
    selected_fs_path = dirs['tables'] / 'selected_feature_set.json'

    if is_multi_fs and multi_fs_path.exists():
        multi_fs_info = load_json(multi_fs_path)
        fs_results = multi_fs_info.get('fs_results', {})
        if champion_fs in fs_results:
            selected_features = fs_results[champion_fs].get('features', [])
    elif selected_fs_path.exists():
        fs_info = load_json(selected_fs_path)
        selected_features = fs_info.get('selected_features', [])

    # Collect top configurations info for comparison
    top_configs_details = []
    id_col = 'model_id' if is_multi_fs else 'model'
    for model_id in top_model_ids:
        model_row = ranking_df[ranking_df[id_col] == model_id]
        if not model_row.empty:
            row_dict = model_row.iloc[0].to_dict()
            config_detail = {
                'model_id': model_id,
                'score': float(row_dict[score_col]),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in row_dict.items()
                           if not k.startswith('vikor_') and not k.startswith('topsis_')
                           and k not in ['model', 'model_id']}
            }
            if is_multi_fs:
                config_detail['fs_option'] = row_dict.get('fs_option')
                config_detail['model'] = row_dict.get('model')
            top_configs_details.append(config_detail)

    # Build champion pipeline record
    champion_pipeline = {
        'champion_model': champion_model,
        'champion_model_id': champion_model_id,
        'champion_fs_option': champion_fs,
        'mcda_method': config.mcda.method,
        'mcda_score': float(champion_score),
        'best_params': best_params,
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'top_configurations': top_model_ids,
        'top_configurations_details': top_configs_details,
        'metrics': {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in champion_metrics.items()
            if not k.startswith('vikor_') and not k.startswith('topsis_')
            and k not in ['model', 'model_id']
        },
        'criteria': criteria,
        'weights': weights,
        'is_multi_fs': is_multi_fs
    }

    # Add full ranking
    rank_cols = [id_col, score_col, rank_col]
    if is_multi_fs:
        rank_cols = ['model_id', 'fs_option', 'model', score_col, rank_col]
    champion_pipeline['ranking'] = ranking_df[rank_cols].to_dict(orient='records')

    save_json_numpy(champion_pipeline, dirs['tables'] / 'champion_pipeline.json')

    # Save full FS-model comparison table
    if is_multi_fs:
        fs_model_full = metrics_df.copy()
        fs_model_full.to_csv(dirs['tables'] / 'fs_model_full_comparison.csv', index=False)
        logger.info("Saved full FS-model comparison table")

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
            title="Model Selection: Pareto Front (All FS Options)" if is_multi_fs else "Model Selection: Pareto Front",
            output_path=dirs['figures'] / 'pareto_front.png'
        )

    # MCDA ranking plot (use model_id for multi-FS)
    ranking_for_plot = ranking_df.copy()
    if is_multi_fs:
        # Use model_id as the label
        ranking_for_plot['model'] = ranking_for_plot['model_id']

    plot_mcda_ranking(
        ranking_for_plot,
        score_col=score_col,
        title=f"Model Ranking ({config.mcda.method.upper()})" + (" - All FS Options" if is_multi_fs else ""),
        output_path=dirs['figures'] / 'mcda_model_ranking.png'
    )

    # =========================================
    # Multi-FS Comparison Plots
    # =========================================
    if is_multi_fs:
        logger.info("-" * 40)
        logger.info("Generating multi-FS comparison plots...")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Heatmap: Models vs FS Options (weighted_mae)
            heatmap_data = metrics_df.pivot_table(
                index='model', columns='fs_option', values='weighted_mae', aggfunc='first'
            )
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax)
            ax.set_title('Model Performance Across FS Options (Weighted MAE - Lower is Better)')
            ax.set_xlabel('Feature Selection Option')
            ax.set_ylabel('Model')
            plt.tight_layout()
            plt.savefig(dirs['figures'] / 'fs_model_heatmap.png', dpi=config.output.figure_dpi)
            plt.close()
            logger.info("Generated FS-model heatmap")

            # Bar chart: Best model per FS option
            best_per_fs = metrics_df.loc[metrics_df.groupby('fs_option')['weighted_mae'].idxmin()]
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(best_per_fs['fs_option'], best_per_fs['weighted_mae'], color='steelblue')
            ax.set_xlabel('Feature Selection Option')
            ax.set_ylabel('Weighted MAE')
            ax.set_title('Best Model Performance per FS Option')

            # Annotate with model names
            for bar, model_name in zip(bars, best_per_fs['model']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       model_name, ha='center', va='bottom', fontsize=9, rotation=45)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(dirs['figures'] / 'best_per_fs_option.png', dpi=config.output.figure_dpi)
            plt.close()
            logger.info("Generated best-per-FS bar chart")

            # Grouped bar chart: All models across FS options
            fig, ax = plt.subplots(figsize=(14, 8))
            models = metrics_df['model'].unique()
            fs_opts = metrics_df['fs_option'].unique()
            x = np.arange(len(fs_opts))
            width = 0.8 / len(models)

            for i, model in enumerate(models):
                model_data = metrics_df[metrics_df['model'] == model]
                values = [model_data[model_data['fs_option'] == fs]['weighted_mae'].values[0]
                         if len(model_data[model_data['fs_option'] == fs]) > 0 else np.nan
                         for fs in fs_opts]
                ax.bar(x + i * width, values, width, label=model)

            ax.set_xlabel('Feature Selection Option')
            ax.set_ylabel('Weighted MAE')
            ax.set_title('Model Comparison Across All FS Options')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(fs_opts, rotation=45, ha='right')
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(dirs['figures'] / 'all_models_fs_comparison.png', dpi=config.output.figure_dpi)
            plt.close()
            logger.info("Generated all-models FS comparison")

            # Comprehensive multi-FS summary (4-panel plot)
            from src.reporting.plots import plot_multi_fs_summary
            plot_multi_fs_summary(
                metrics_df,
                output_path=dirs['figures'] / 'multi_fs_summary.png'
            )
            logger.info("Generated comprehensive multi-FS summary plot")

        except Exception as e:
            logger.warning(f"Multi-FS comparison plots failed: {e}")

    # =========================================
    # VIKOR Radar Chart
    # =========================================
    logger.info("-" * 40)
    logger.info("Generating VIKOR radar chart...")

    try:
        from src.reporting.plots import plot_vikor_radar

        plot_vikor_radar(
            ranking_df,
            criteria=criteria,
            top_n=min(5, len(ranking_df)),
            title=f"VIKOR Model Comparison (Radar Chart)",
            output_path=dirs['figures'] / 'vikor_radar_chart.png'
        )
        logger.info("Generated VIKOR radar chart")
    except Exception as e:
        logger.warning(f"Radar chart failed: {e}")

    # =========================================
    # Sensitivity Analysis
    # =========================================
    logger.info("-" * 40)
    logger.info("Running sensitivity analysis...")

    try:
        from src.decision.sensitivity import vikor_v_sensitivity, weight_sensitivity
        from src.reporting.plots import plot_vikor_v_sensitivity, plot_weight_sensitivity_heatmap

        # V parameter sensitivity (VIKOR only)
        if config.mcda.method == 'vikor':
            v_sensitivity = vikor_v_sensitivity(
                pareto_df.copy(), criteria, weights, criteria_types
            )
            v_sensitivity.to_csv(dirs['tables'] / 'vikor_v_sensitivity.csv', index=False)

            plot_vikor_v_sensitivity(
                v_sensitivity,
                title="VIKOR Ranking Sensitivity to v Parameter",
                output_path=dirs['figures'] / 'sensitivity_v_parameter.png'
            )
            logger.info("Generated v-parameter sensitivity plot")

        # Weight perturbation sensitivity
        weight_sens = weight_sensitivity(
            pareto_df.copy(), criteria, weights, criteria_types,
            method=config.mcda.method, perturbation=0.2
        )
        weight_sens.to_csv(dirs['tables'] / 'weight_sensitivity.csv', index=False)

        plot_weight_sensitivity_heatmap(
            weight_sens,
            title="Rank Stability Under Weight Perturbations (+/- 20%)",
            output_path=dirs['figures'] / 'sensitivity_weights.png'
        )
        logger.info("Generated weight sensitivity heatmap")

    except Exception as e:
        logger.warning(f"Sensitivity analysis failed: {e}")

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Model Selection Complete!")

    if is_multi_fs:
        logger.info(f"  Mode: Multi-FS comparison ({len(fs_options)} FS options)")
        logger.info(f"  Total configurations evaluated: {len(metrics_df)}")
        logger.info(f"  Pareto front size: {len(pareto_df)}")
        logger.info(f"  Champion configuration: {champion_model_id}")
        logger.info(f"    FS option: {champion_fs}")
        logger.info(f"    Model: {champion_model}")
        logger.info(f"    MCDA score: {champion_score:.4f}")

        # Show top configurations
        logger.info(f"  Top {len(top_model_ids)} configurations:")
        for i, config_id in enumerate(top_model_ids):
            row = ranking_df[ranking_df['model_id'] == config_id].iloc[0]
            logger.info(f"    {i+1}. {config_id} (score: {row[score_col]:.4f})")
    else:
        logger.info(f"  Mode: Single-FS")
        logger.info(f"  Champion model: {champion_model}")
        logger.info(f"  MCDA method: {config.mcda.method}")
        logger.info(f"  MCDA score: {champion_score:.4f}")
        if len(top_model_ids) > 1:
            second_model = top_model_ids[1]
            second_row = ranking_df[ranking_df['model'] == second_model]
            if not second_row.empty:
                logger.info(f"  Second best: {second_model} (score: {second_row.iloc[0][score_col]:.4f})")
        logger.info(f"  Pareto front size: {len(pareto_df)}")

    logger.info(f"  Results saved to: {dirs['tables']}")
    logger.info(f"  Figures saved to: {dirs['figures']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
