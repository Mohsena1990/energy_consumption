"""
Traditional ML models for CO2 forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster, ModelRegistry
from ..core.logging_utils import get_logger


@ModelRegistry.register('ridge')
class RidgeModel(BaseForecaster):
    """Ridge regression model."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'normalize': False
        }
        params = {**default_params, **(params or {})}
        super().__init__('ridge', params)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeModel':
        logger = get_logger()

        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model = SklearnRidge(
            alpha=self.params.get('alpha', 1.0),
            fit_intercept=self.params.get('fit_intercept', True)
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        logger.debug(f"Ridge fitted with alpha={self.params['alpha']}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or self.feature_names is None:
            return None

        # Use absolute coefficients as importance
        importances = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, importances))

    @property
    def supports_shap(self) -> bool:
        return False  # Use permutation importance instead

    @property
    def interpretability_score(self) -> float:
        return 0.9  # Linear models are highly interpretable


@ModelRegistry.register('random_forest')
class RandomForestModel(BaseForecaster):
    """Random Forest regression model."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **(params or {})}
        super().__init__('random_forest', params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        logger = get_logger()

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

        logger.debug(f"Random Forest fitted with {self.params['n_estimators']} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or self.feature_names is None:
            return None

        return dict(zip(self.feature_names, self.model.feature_importances_))

    @property
    def supports_shap(self) -> bool:
        return True

    @property
    def interpretability_score(self) -> float:
        return 0.6


@ModelRegistry.register('lightgbm')
class LightGBMModel(BaseForecaster):
    """LightGBM regression model."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'verbosity': -1
        }
        params = {**default_params, **(params or {})}
        super().__init__('lightgbm', params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMModel':
        logger = get_logger()

        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

        logger.debug(f"LightGBM fitted with {self.params['n_estimators']} iterations")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or self.feature_names is None:
            return None

        return dict(zip(self.feature_names, self.model.feature_importances_))

    @property
    def supports_shap(self) -> bool:
        return True

    @property
    def interpretability_score(self) -> float:
        return 0.7


@ModelRegistry.register('catboost')
class CatBoostModel(BaseForecaster):
    """CatBoost regression model."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        params = {**default_params, **(params or {})}
        super().__init__('catboost', params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostModel':
        logger = get_logger()

        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost is required. Install with: pip install catboost")

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values

        self.model = CatBoostRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

        logger.debug(f"CatBoost fitted with {self.params['iterations']} iterations")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or self.feature_names is None:
            return None

        return dict(zip(self.feature_names, self.model.feature_importances_))

    @property
    def supports_shap(self) -> bool:
        return True

    @property
    def interpretability_score(self) -> float:
        return 0.7


def get_model_param_space(model_name: str) -> Dict[str, Any]:
    """
    Get parameter search space for optimization.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with parameter bounds
    """
    param_spaces = {
        'ridge': {
            'alpha': {'type': 'log_uniform', 'low': 0.001, 'high': 100.0}
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'learning_rate': {'type': 'log_uniform', 'low': 0.01, 'high': 0.3},
            'num_leaves': {'type': 'int', 'low': 15, 'high': 127},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 50},
            'reg_alpha': {'type': 'log_uniform', 'low': 0.001, 'high': 10.0},
            'reg_lambda': {'type': 'log_uniform', 'low': 0.001, 'high': 10.0}
        },
        'catboost': {
            'iterations': {'type': 'int', 'low': 50, 'high': 300},
            'learning_rate': {'type': 'log_uniform', 'low': 0.01, 'high': 0.3},
            'depth': {'type': 'int', 'low': 3, 'high': 10},
            'l2_leaf_reg': {'type': 'log_uniform', 'low': 0.1, 'high': 10.0}
        }
    }

    return param_spaces.get(model_name, {})
