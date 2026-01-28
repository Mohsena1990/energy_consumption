"""
Base model interface for CO2 forecasting framework.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'BaseForecaster':
        """Set model parameters."""
        self.params.update(params)
        return self

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> 'BaseForecaster':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return None

    @property
    def supports_shap(self) -> bool:
        """Whether this model supports SHAP explanations."""
        return False

    @property
    def interpretability_score(self) -> float:
        """Interpretability score (0-1, higher is more interpretable)."""
        return 0.5

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class ModelRegistry:
    """Registry for available forecasting models."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def create(cls, name: str, params: Dict[str, Any] = None) -> BaseForecaster:
        """Create a model instance by name."""
        model_class = cls.get(name)
        return model_class(params=params)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())
