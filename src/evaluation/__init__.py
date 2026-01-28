"""
Evaluation module for CO2 forecasting framework.
"""
from .metrics import (
    ForecastMetrics,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    calculate_bias,
    compute_all_metrics,
    evaluate_by_horizon,
    evaluate_stability,
    create_evaluation_summary,
    compare_models
)

__all__ = [
    'ForecastMetrics',
    'calculate_mae', 'calculate_rmse', 'calculate_mape',
    'calculate_r2', 'calculate_bias',
    'compute_all_metrics', 'evaluate_by_horizon', 'evaluate_stability',
    'create_evaluation_summary', 'compare_models'
]
