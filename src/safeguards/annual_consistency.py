"""
Annual consistency safeguards for CO2 forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from ..core.logging_utils import get_logger
from ..core.utils import inverse_log_transform, aggregate_to_annual
from ..evaluation.metrics import calculate_mae, calculate_mape, calculate_bias


def aggregate_quarterly_to_annual(
    predictions_df: pd.DataFrame,
    transform: str = 'log',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Aggregate quarterly predictions to annual totals.

    Args:
        predictions_df: DataFrame with quarterly predictions
        transform: Target transformation used ('log' or 'delta_log')
        date_col: Column containing dates

    Returns:
        DataFrame with annual aggregations
    """
    logger = get_logger()

    df = predictions_df.copy()

    # Convert predictions back to original scale if log-transformed
    if transform == 'log':
        df['actual_original'] = inverse_log_transform(df['actual'].values)
        df['predicted_original'] = inverse_log_transform(df['predicted'].values)
    else:
        df['actual_original'] = df['actual']
        df['predicted_original'] = df['predicted']

    # Extract year from date
    if date_col in df.columns:
        df['year'] = pd.to_datetime(df[date_col]).dt.year
    elif isinstance(df.index, pd.DatetimeIndex):
        df['year'] = df.index.year
    else:
        logger.warning("Could not extract year from date, using index")
        df['year'] = df.index

    # Aggregate to annual
    annual = df.groupby('year').agg({
        'actual_original': 'sum',
        'predicted_original': 'sum'
    }).reset_index()

    annual.columns = ['year', 'actual_annual', 'predicted_annual']

    return annual


def check_annual_consistency(
    predictions_df: pd.DataFrame,
    transform: str = 'log'
) -> Dict[str, Any]:
    """
    Check annual consistency of quarterly predictions.

    Args:
        predictions_df: DataFrame with quarterly predictions
        transform: Target transformation used

    Returns:
        Dictionary with consistency metrics
    """
    logger = get_logger()

    # Aggregate to annual
    annual = aggregate_quarterly_to_annual(predictions_df, transform)

    if len(annual) == 0:
        logger.warning("No annual data available for consistency check")
        return {'error': 'No data'}

    # Calculate annual metrics
    y_true = annual['actual_annual'].values
    y_pred = annual['predicted_annual'].values

    results = {
        'annual_mae': calculate_mae(y_true, y_pred),
        'annual_mape': calculate_mape(y_true, y_pred),
        'annual_bias': calculate_bias(y_true, y_pred),
        'annual_bias_pct': (calculate_bias(y_true, y_pred) / np.mean(y_true)) * 100,
        'n_years': len(annual),
        'annual_data': annual.to_dict(orient='records')
    }

    # Calculate year-by-year errors
    annual['error'] = annual['actual_annual'] - annual['predicted_annual']
    annual['error_pct'] = (annual['error'] / annual['actual_annual']) * 100

    results['year_errors'] = annual[['year', 'error', 'error_pct']].to_dict(orient='records')

    # Consistency score (1 - normalized MAPE)
    results['consistency_score'] = max(0, 1 - results['annual_mape'] / 100)

    logger.info(f"Annual consistency: MAE={results['annual_mae']:.2f}, MAPE={results['annual_mape']:.2f}%")

    return results


def create_annual_baseline(
    y_annual: pd.Series,
    method: str = 'naive'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create simple annual baseline predictions.

    Args:
        y_annual: Annual target values
        method: Baseline method ('naive', 'mean', 'linear', 'arima')

    Returns:
        Tuple of (predictions, baseline info)
    """
    logger = get_logger()

    n = len(y_annual)
    if n < 2:
        return np.array([]), {'error': 'Not enough data'}

    if method == 'naive':
        # Naive: use last year's value
        predictions = np.roll(y_annual.values, 1)
        predictions[0] = y_annual.values[0]

    elif method == 'mean':
        # Mean: use expanding mean
        predictions = np.array([
            y_annual.values[:i].mean() if i > 0 else y_annual.values[0]
            for i in range(n)
        ])

    elif method == 'linear':
        # Linear trend
        from sklearn.linear_model import LinearRegression
        X = np.arange(n).reshape(-1, 1)
        y = y_annual.values

        # Leave-one-out predictions
        predictions = np.zeros(n)
        for i in range(n):
            if i == 0:
                predictions[i] = y[0]
            else:
                model = LinearRegression()
                model.fit(X[:i], y[:i])
                predictions[i] = model.predict(X[i:i+1])[0]

    elif method == 'arima':
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Fit ARIMA(1,1,0) as simple baseline
            predictions = np.zeros(n)
            for i in range(n):
                if i < 3:
                    predictions[i] = y_annual.values[max(0, i-1)]
                else:
                    try:
                        model = ARIMA(y_annual.values[:i], order=(1, 1, 0))
                        fit = model.fit()
                        predictions[i] = fit.forecast(1)[0]
                    except:
                        predictions[i] = y_annual.values[i-1]

        except ImportError:
            logger.warning("statsmodels not available, using naive baseline")
            return create_annual_baseline(y_annual, 'naive')

    else:
        raise ValueError(f"Unknown baseline method: {method}")

    info = {
        'method': method,
        'n_years': n
    }

    return predictions, info


def compare_with_annual_baseline(
    predictions_df: pd.DataFrame,
    y_annual_true: pd.Series,
    transform: str = 'log',
    baseline_methods: List[str] = ['naive', 'mean', 'linear']
) -> Dict[str, Any]:
    """
    Compare model predictions with annual baselines.

    Args:
        predictions_df: DataFrame with quarterly model predictions
        y_annual_true: True annual values
        transform: Target transformation
        baseline_methods: List of baseline methods to compare

    Returns:
        Comparison results
    """
    logger = get_logger()

    # Get model's annual predictions
    model_annual = aggregate_quarterly_to_annual(predictions_df, transform)

    results = {
        'model': {},
        'baselines': {}
    }

    # Align data by year
    common_years = set(model_annual['year'].values) & set(y_annual_true.index)
    common_years = sorted(list(common_years))

    if len(common_years) == 0:
        logger.warning("No common years for comparison")
        return {'error': 'No common years'}

    # Model metrics
    model_pred = model_annual[model_annual['year'].isin(common_years)]['predicted_annual'].values
    actual = y_annual_true.loc[common_years].values

    results['model'] = {
        'mae': calculate_mae(actual, model_pred),
        'mape': calculate_mape(actual, model_pred),
        'bias': calculate_bias(actual, model_pred)
    }

    # Baseline metrics
    for method in baseline_methods:
        baseline_pred, _ = create_annual_baseline(y_annual_true, method)

        # Align baseline predictions
        if len(baseline_pred) > 0:
            baseline_df = pd.DataFrame({
                'year': y_annual_true.index,
                'predicted': baseline_pred
            })
            baseline_aligned = baseline_df[baseline_df['year'].isin(common_years)]['predicted'].values

            results['baselines'][method] = {
                'mae': calculate_mae(actual, baseline_aligned),
                'mape': calculate_mape(actual, baseline_aligned),
                'bias': calculate_bias(actual, baseline_aligned)
            }

    # Determine if model beats baselines
    model_mae = results['model']['mae']
    baseline_maes = [b['mae'] for b in results['baselines'].values()]

    if baseline_maes:
        results['beats_all_baselines'] = all(model_mae < b_mae for b_mae in baseline_maes)
        results['best_baseline'] = min(results['baselines'].items(), key=lambda x: x[1]['mae'])[0]
        results['improvement_over_best'] = min(baseline_maes) - model_mae
    else:
        results['beats_all_baselines'] = None
        results['best_baseline'] = None
        results['improvement_over_best'] = None

    logger.info(f"Model MAE: {model_mae:.2f}, Best baseline: {results.get('best_baseline')}")

    return results
