"""
Evaluation metrics for CO2 forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from ..core.logging_utils import get_logger
from ..core.utils import calculate_weighted_mae, inverse_log_transform, aggregate_to_annual


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    mse: float = 0.0
    r2: float = 0.0
    bias: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'mse': self.mse,
            'r2': self.r2,
            'bias': self.bias,
            'n_samples': self.n_samples
        }


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias (systematic error)."""
    return np.mean(y_pred - y_true)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> ForecastMetrics:
    """
    Compute all forecast metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        ForecastMetrics object
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have same length")

    if len(y_true) == 0:
        return ForecastMetrics()

    return ForecastMetrics(
        mae=calculate_mae(y_true, y_pred),
        rmse=calculate_rmse(y_true, y_pred),
        mape=calculate_mape(y_true, y_pred),
        mse=np.mean((y_true - y_pred) ** 2),
        r2=calculate_r2(y_true, y_pred),
        bias=calculate_bias(y_true, y_pred),
        n_samples=len(y_true)
    )


def evaluate_by_horizon(
    predictions_df: pd.DataFrame,
    horizons: List[int]
) -> Dict[int, ForecastMetrics]:
    """
    Evaluate metrics by forecast horizon.

    Args:
        predictions_df: DataFrame with columns ['horizon', 'actual', 'predicted']
        horizons: List of horizons to evaluate

    Returns:
        Dictionary mapping horizon to metrics
    """
    results = {}

    for h in horizons:
        h_data = predictions_df[predictions_df['horizon'] == h]

        if len(h_data) == 0:
            results[h] = ForecastMetrics()
            continue

        y_true = h_data['actual'].values
        y_pred = h_data['predicted'].values

        results[h] = compute_all_metrics(y_true, y_pred)

    return results


def evaluate_stability(
    predictions_df: pd.DataFrame,
    group_col: str = 'fold_id'
) -> Dict[str, float]:
    """
    Evaluate prediction stability across folds.

    Args:
        predictions_df: DataFrame with predictions
        group_col: Column to group by (e.g., 'fold_id')

    Returns:
        Stability metrics
    """
    if group_col not in predictions_df.columns:
        return {}

    # Calculate error per fold
    fold_errors = []
    for fold_id, group in predictions_df.groupby(group_col):
        y_true = group['actual'].values
        y_pred = group['predicted'].values
        mae = calculate_mae(y_true, y_pred)
        fold_errors.append(mae)

    if not fold_errors:
        return {}

    return {
        'error_mean': np.mean(fold_errors),
        'error_std': np.std(fold_errors),
        'error_min': np.min(fold_errors),
        'error_max': np.max(fold_errors),
        'error_range': np.max(fold_errors) - np.min(fold_errors),
        'stability_score': 1 / (1 + np.std(fold_errors)),  # Higher is better
        'worst_fold_error': np.max(fold_errors),
        'n_folds': len(fold_errors)
    }


def create_evaluation_summary(
    model_name: str,
    horizon_metrics: Dict[int, ForecastMetrics],
    stability_metrics: Dict[str, float],
    horizon_weights: Dict[int, float]
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation summary.

    Args:
        model_name: Name of the model
        horizon_metrics: Metrics by horizon
        stability_metrics: Stability metrics
        horizon_weights: Weights for each horizon

    Returns:
        Evaluation summary dictionary
    """
    # Calculate weighted MAE
    mae_by_horizon = {h: m.mae for h, m in horizon_metrics.items()}
    weighted_mae = calculate_weighted_mae(mae_by_horizon, horizon_weights)

    summary = {
        'model': model_name,
        'weighted_mae': weighted_mae,
        'stability_score': stability_metrics.get('stability_score', 0),
        'error_std': stability_metrics.get('error_std', 0),
        'worst_fold_error': stability_metrics.get('worst_fold_error', 0)
    }

    # Add per-horizon metrics
    for h, metrics in horizon_metrics.items():
        summary[f'mae_h{h}'] = metrics.mae
        summary[f'rmse_h{h}'] = metrics.rmse
        summary[f'mape_h{h}'] = metrics.mape
        summary[f'r2_h{h}'] = metrics.r2

    return summary


def compare_models(
    model_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create comparison table of all models.

    Args:
        model_results: Dictionary of model evaluation results

    Returns:
        Comparison DataFrame
    """
    rows = []

    for model_name, results in model_results.items():
        row = {'model': model_name}
        row.update({k: v for k, v in results.items() if k != 'model'})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by weighted MAE
    if 'weighted_mae' in df.columns:
        df = df.sort_values('weighted_mae')

    return df
