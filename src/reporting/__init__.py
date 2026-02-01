"""
Reporting module for CO2 forecasting framework.
High-resolution plotting and visualization with bigger, bolder text.
"""
from .plots import (
    set_plot_style,
    plot_predictions_vs_actual,
    plot_horizon_comparison,
    plot_model_comparison,
    plot_optimization_history,
    plot_pareto_front,
    plot_mcda_ranking,
    plot_feature_importance,
    plot_shap_summary,
    plot_annual_consistency,
    plot_regime_comparison,
    create_all_plots,
    # New enhanced plots
    plot_shap_beeswarm,
    plot_model_coefficients,
    plot_vikor_radar,
    plot_vikor_v_sensitivity,
    plot_weight_sensitivity_heatmap,
    plot_pareto_front_enhanced,
    # Multi-FS comparison plots
    plot_fs_model_comparison_heatmap,
    plot_fs_comparison_bars,
    plot_best_per_fs_option,
    plot_multi_fs_summary
)

__all__ = [
    'set_plot_style',
    'plot_predictions_vs_actual',
    'plot_horizon_comparison',
    'plot_model_comparison',
    'plot_optimization_history',
    'plot_pareto_front',
    'plot_mcda_ranking',
    'plot_feature_importance',
    'plot_shap_summary',
    'plot_annual_consistency',
    'plot_regime_comparison',
    'create_all_plots',
    # New enhanced plots
    'plot_shap_beeswarm',
    'plot_model_coefficients',
    'plot_vikor_radar',
    'plot_vikor_v_sensitivity',
    'plot_weight_sensitivity_heatmap',
    'plot_pareto_front_enhanced',
    # Multi-FS comparison plots
    'plot_fs_model_comparison_heatmap',
    'plot_fs_comparison_bars',
    'plot_best_per_fs_option',
    'plot_multi_fs_summary'
]
