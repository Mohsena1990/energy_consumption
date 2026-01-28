"""
CO2 Forecasting Framework
=========================

A modular framework for quarterly CO2 emissions forecasting with:
- Multiple feature selection strategies (linear, nonlinear, consensus)
- SHAP-based feature selection evaluation
- MCDA (VIKOR/TOPSIS) for decision making
- Walk-forward cross-validation
- Swarm optimization (PSO/GWO) for hyperparameter tuning
- Annual consistency safeguards
- High-resolution visualization

Modules:
    core: Configuration and utilities
    data_io: Data loading and schema mapping
    quality: Data quality auditing
    features: Feature engineering
    splits: Walk-forward CV splits
    fs: Feature selection methods
    models: Forecasting models
    optimization: Swarm optimization
    evaluation: Metrics and evaluation
    safeguards: Annual consistency checks
    decision: MCDA methods (VIKOR, TOPSIS)
    interpretability: SHAP analysis
    reporting: Visualization and plotting
"""

__version__ = "1.0.0"
__author__ = "CO2 Forecasting Team"

from . import core
from . import data_io
from . import quality
from . import features
from . import splits
from . import fs
from . import models
from . import optimization
from . import evaluation
from . import safeguards
from . import decision
from . import interpretability
from . import reporting
