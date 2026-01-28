"""
Decision making module for CO2 forecasting framework.
MCDA methods: VIKOR and TOPSIS.
"""
from .mcda import (
    normalize_matrix,
    pareto_filter,
    topsis,
    vikor,
    select_best_fs_option,
    select_best_model
)

__all__ = [
    'normalize_matrix',
    'pareto_filter',
    'topsis',
    'vikor',
    'select_best_fs_option',
    'select_best_model'
]
