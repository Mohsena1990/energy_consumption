"""
Feature selection module for CO2 forecasting framework.
"""
from .vif import calculate_vif, filter_by_vif
from .linear import (
    ridge_stability_selection,
    elasticnet_stability_selection,
    fs_linear
)
from .nonlinear import (
    random_forest_importance,
    lightgbm_importance,
    catboost_importance,
    boruta_selection,
    fs_nonlinear
)
from .consensus import (
    vote_based_selection,
    stability_based_selection,
    fs_consensus,
    run_all_fs_options
)
from .evaluation import (
    evaluate_fs_option_with_shap,
    create_fs_evaluation_matrix
)

__all__ = [
    'calculate_vif', 'filter_by_vif',
    'ridge_stability_selection', 'elasticnet_stability_selection', 'fs_linear',
    'random_forest_importance', 'lightgbm_importance', 'catboost_importance',
    'boruta_selection', 'fs_nonlinear',
    'vote_based_selection', 'stability_based_selection', 'fs_consensus',
    'run_all_fs_options',
    'evaluate_fs_option_with_shap', 'create_fs_evaluation_matrix'
]
