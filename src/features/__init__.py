"""
Feature engineering module for CO2 forecasting framework.
"""
from .engineering import (
    create_target_variable,
    create_lag_features,
    create_seasonality_features,
    create_shock_features,
    create_rolling_features,
    engineer_features,
    create_feature_dictionary
)

__all__ = [
    'create_target_variable',
    'create_lag_features',
    'create_seasonality_features',
    'create_shock_features',
    'create_rolling_features',
    'engineer_features',
    'create_feature_dictionary'
]
