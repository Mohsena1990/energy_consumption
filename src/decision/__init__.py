"""
Decision making module for CO2 forecasting framework.
MCDA methods: VIKOR and TOPSIS with sensitivity analysis.
"""
from .mcda import (
    normalize_matrix,
    pareto_filter,
    topsis,
    vikor,
    select_best_fs_option,
    select_best_model
)
from .sensitivity import (
    vikor_v_sensitivity,
    weight_sensitivity,
    criterion_removal_sensitivity,
    compute_rank_stability_score
)

__all__ = [
    'normalize_matrix',
    'pareto_filter',
    'topsis',
    'vikor',
    'select_best_fs_option',
    'select_best_model',
    # Sensitivity analysis
    'vikor_v_sensitivity',
    'weight_sensitivity',
    'criterion_removal_sensitivity',
    'compute_rank_stability_score'
]
