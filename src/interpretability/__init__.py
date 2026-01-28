"""
Interpretability module for CO2 forecasting framework.
SHAP analysis and feature importance.
"""
from .shap_analysis import (
    compute_shap_values,
    get_feature_importance_from_shap,
    analyze_regime_shap,
    analyze_seasonal_shap,
    compute_permutation_importance,
    get_ridge_coefficients,
    generate_interpretation_report
)

__all__ = [
    'compute_shap_values',
    'get_feature_importance_from_shap',
    'analyze_regime_shap',
    'analyze_seasonal_shap',
    'compute_permutation_importance',
    'get_ridge_coefficients',
    'generate_interpretation_report'
]
