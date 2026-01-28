"""
Reporting module for CO2 forecasting framework.
High-resolution plotting and visualization.
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
    create_all_plots
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
    'create_all_plots'
]
