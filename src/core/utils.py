"""
General utilities for CO2 forecasting framework.
"""
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]):
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def hash_dict(d: Dict) -> str:
    """Create a hash of a dictionary for caching/identification."""
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def quarter_to_date(quarter_str: str) -> pd.Timestamp:
    """
    Convert quarter string to timestamp.
    Handles formats: '1999Q1', '1999-Q1', '1999 Q1'
    """
    quarter_str = str(quarter_str).strip().replace('-', '').replace(' ', '')
    try:
        return pd.Period(quarter_str, freq='Q').to_timestamp()
    except:
        # Try other formats
        if 'Q' in quarter_str:
            year = int(quarter_str[:4])
            q = int(quarter_str[-1])
            month = (q - 1) * 3 + 1
            return pd.Timestamp(year=year, month=month, day=1)
        raise ValueError(f"Cannot parse quarter: {quarter_str}")


def date_to_quarter(date: pd.Timestamp) -> str:
    """Convert timestamp to quarter string."""
    return f"{date.year}Q{date.quarter}"


def ensure_quarterly_index(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Ensure DataFrame has a quarterly datetime index.
    """
    df = df.copy()

    if date_col is not None and date_col in df.columns:
        df[date_col] = df[date_col].apply(quarter_to_date)
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index
        df.index = df.index.map(quarter_to_date)

    df.index = pd.DatetimeIndex(df.index, freq='QS')
    df.index.name = 'date'
    return df


def calculate_weighted_mae(
    errors_by_horizon: Dict[int, float],
    weights: Dict[int, float]
) -> float:
    """
    Calculate weighted MAE across horizons.

    Args:
        errors_by_horizon: MAE for each horizon {1: mae1, 2: mae2, 4: mae4}
        weights: Weights for each horizon {1: 0.5, 2: 0.3, 4: 0.2}

    Returns:
        Weighted MAE
    """
    total = 0.0
    for h, mae in errors_by_horizon.items():
        w = weights.get(h, 0.0)
        total += w * mae
    return total


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Inverse of log transform."""
    return np.exp(y_log)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return a / b if b != 0 else default


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_year_from_quarter(date: pd.Timestamp) -> int:
    """Get year from quarterly timestamp."""
    return date.year


def aggregate_to_annual(
    quarterly_values: pd.Series,
    agg_func: str = 'sum'
) -> pd.Series:
    """Aggregate quarterly values to annual."""
    annual = quarterly_values.groupby(quarterly_values.index.year)
    if agg_func == 'sum':
        return annual.sum()
    elif agg_func == 'mean':
        return annual.mean()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


def save_json_numpy(data: Any, path: Union[str, Path], indent: int = 2):
    """Save data to JSON file with numpy support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
