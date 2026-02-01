"""
Feature engineering for CO2 forecasting framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.config import Config, FeatureConfig
from ..core.logging_utils import get_logger


def create_target_variable(
    df: pd.DataFrame,
    target_col: str,
    transform: str = 'log'
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Create target variable with transformation.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        transform: Transformation type ('log' or 'delta_log')

    Returns:
        Tuple of (transformed target series, transform metadata)
    """
    logger = get_logger()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y_raw = df[target_col].copy()

    metadata = {
        'original_column': target_col,
        'transform': transform,
        'original_min': float(y_raw.min()),
        'original_max': float(y_raw.max()),
        'original_mean': float(y_raw.mean())
    }

    if transform == 'log':
        # Simple log transform
        y = np.log(y_raw)
        logger.info(f"Applied log transform to target. Range: {y.min():.4f} to {y.max():.4f}")

    elif transform == 'delta_log':
        # Log difference (growth rate)
        y = np.log(y_raw).diff()
        logger.info(f"Applied delta-log transform to target. Range: {y.min():.4f} to {y.max():.4f}")

    else:
        # No transform
        y = y_raw
        logger.info(f"No transform applied to target")

    metadata['transformed_min'] = float(y.min()) if not pd.isna(y.min()) else None
    metadata['transformed_max'] = float(y.max()) if not pd.isna(y.max()) else None

    return y, metadata


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    prefix: str = 'lag'
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag orders (e.g., [1, 2, 3, 4])
        prefix: Prefix for lag column names

    Returns:
        DataFrame with lag features
    """
    logger = get_logger()
    lag_features = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping lag creation")
            continue

        for lag in lags:
            lag_col_name = f"{col}_{prefix}{lag}"
            lag_features[lag_col_name] = df[col].shift(lag)

    lag_df = pd.DataFrame(lag_features, index=df.index)
    logger.info(f"Created {len(lag_features)} lag features")

    return lag_df


def create_seasonality_features(
    df: pd.DataFrame,
    method: str = 'dummies'
) -> pd.DataFrame:
    """
    Create seasonality features.

    Args:
        df: DataFrame with datetime index
        method: 'dummies' for quarter dummies, 'sincos' for sine/cosine encoding

    Returns:
        DataFrame with seasonality features
    """
    logger = get_logger()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for seasonality features")

    quarter = df.index.quarter

    if method == 'dummies':
        # Quarter dummy variables (Q1 as reference)
        season_df = pd.DataFrame(index=df.index)
        for q in [2, 3, 4]:
            season_df[f'Q{q}'] = (quarter == q).astype(int)
        logger.info("Created quarter dummy features (Q2, Q3, Q4)")

    elif method == 'sincos':
        # Sine/cosine encoding
        # Quarter 1-4 maps to angle 0 to 2*pi*(3/4)
        angle = 2 * np.pi * (quarter - 1) / 4
        season_df = pd.DataFrame({
            'season_sin': np.sin(angle),
            'season_cos': np.cos(angle)
        }, index=df.index)
        logger.info("Created sine/cosine seasonality features")

    else:
        raise ValueError(f"Unknown seasonality method: {method}")

    return season_df


def create_shock_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Create shock indicator features (COVID, energy crisis, etc.).

    Args:
        df: DataFrame with datetime index
        config: Feature configuration

    Returns:
        DataFrame with shock features
    """
    logger = get_logger()
    shock_df = pd.DataFrame(index=df.index)

    if config.include_covid_dummy:
        # COVID dummy
        from ..core.utils import quarter_to_date
        covid_start = quarter_to_date(config.covid_start)
        covid_end = quarter_to_date(config.covid_end)

        shock_df['COVID'] = ((df.index >= covid_start) & (df.index <= covid_end)).astype(int)
        logger.info(f"Created COVID dummy ({config.covid_start} to {config.covid_end})")

    if config.include_energy_crisis:
        # Energy crisis dummy
        from ..core.utils import quarter_to_date
        crisis_start = quarter_to_date(config.energy_crisis_start)
        crisis_end = quarter_to_date(config.energy_crisis_end)

        shock_df['EnergyCrisis'] = ((df.index >= crisis_start) & (df.index <= crisis_end)).astype(int)
        logger.info(f"Created Energy Crisis dummy ({config.energy_crisis_start} to {config.energy_crisis_end})")

    return shock_df


def create_rate_of_change_features(
    df: pd.DataFrame,
    columns: List[str],
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Create rate-of-change (delta/growth) features.

    Args:
        df: Input DataFrame
        columns: Columns to create rate-of-change features for
        log_transform: If True, use log difference (growth rate); otherwise simple difference

    Returns:
        DataFrame with rate-of-change features
    """
    logger = get_logger()
    roc_features = {}

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping rate-of-change creation")
            continue

        if log_transform:
            # Log difference = growth rate (Δlog)
            # Handle zeros/negatives by adding small constant if needed
            series = df[col].copy()
            if (series <= 0).any():
                series = series + series[series > 0].min() * 0.01
            roc_features[f'{col}_dlog'] = np.log(series).diff()
        else:
            # Simple first difference
            roc_features[f'{col}_diff'] = df[col].diff()

    roc_df = pd.DataFrame(roc_features, index=df.index)
    logger.info(f"Created {len(roc_features)} rate-of-change features")

    return roc_df


def create_intensity_features(
    df: pd.DataFrame,
    target_col: str = 'CO2e',
    denominator_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create intensity/per-capita features.

    Args:
        df: Input DataFrame
        target_col: Target column (numerator)
        denominator_cols: Columns to use as denominators (e.g., Population, TEC)

    Returns:
        DataFrame with intensity features
    """
    logger = get_logger()
    intensity_features = {}

    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return pd.DataFrame(index=df.index)

    # Default denominator columns
    if denominator_cols is None:
        denominator_cols = ['Population', 'TEC']

    for denom_col in denominator_cols:
        if denom_col not in df.columns:
            logger.warning(f"Denominator column '{denom_col}' not found, skipping")
            continue

        # Avoid division by zero
        denom = df[denom_col].replace(0, np.nan)
        intensity_features[f'{target_col}_per_{denom_col}'] = df[target_col] / denom

    intensity_df = pd.DataFrame(intensity_features, index=df.index)
    logger.info(f"Created {len(intensity_features)} intensity features")

    return intensity_df


def create_weather_features(
    df: pd.DataFrame,
    temp_col: str = 'Air_Temp',
    base_temp: float = 18.0
) -> pd.DataFrame:
    """
    Create weather-derived features like Heating Degree Days (HDD).

    Args:
        df: Input DataFrame
        temp_col: Temperature column name
        base_temp: Base temperature for HDD calculation (default 18°C)

    Returns:
        DataFrame with weather features
    """
    logger = get_logger()
    weather_features = {}

    if temp_col not in df.columns:
        logger.warning(f"Temperature column '{temp_col}' not found, skipping weather features")
        return pd.DataFrame(index=df.index)

    # Heating Degree Days proxy: max(0, base_temp - temp)
    weather_features['HDD_proxy'] = np.maximum(0, base_temp - df[temp_col])

    # Cooling Degree Days proxy: max(0, temp - base_temp)
    weather_features['CDD_proxy'] = np.maximum(0, df[temp_col] - base_temp)

    weather_df = pd.DataFrame(weather_features, index=df.index)
    logger.info(f"Created {len(weather_features)} weather features")

    return weather_df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [4, 8],
    functions: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window features.

    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for
        windows: Window sizes
        functions: Aggregation functions

    Returns:
        DataFrame with rolling features
    """
    logger = get_logger()
    rolling_features = {}

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            for func in functions:
                feat_name = f"{col}_roll{window}_{func}"
                if func == 'mean':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).mean()
                elif func == 'std':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).std()
                elif func == 'min':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).min()
                elif func == 'max':
                    rolling_features[feat_name] = df[col].rolling(window=window, min_periods=1).max()

    rolling_df = pd.DataFrame(rolling_features, index=df.index)
    logger.info(f"Created {len(rolling_features)} rolling features")

    return rolling_df


def engineer_features(
    df: pd.DataFrame,
    config: Config,
    target_col: str = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Main feature engineering function.

    Args:
        df: Input DataFrame
        config: Configuration object
        target_col: Target column name (overrides config)

    Returns:
        Tuple of (feature DataFrame, target Series, metadata)
    """
    logger = get_logger()
    logger.info("Starting feature engineering...")

    if target_col is None:
        target_col = config.data.target_column

    metadata = {
        'feature_columns': [],
        'lag_features': [],
        'rolling_features': [],
        'seasonality_features': [],
        'shock_features': [],
        'n_original_features': 0
    }

    # Create target
    y, target_meta = create_target_variable(
        df, target_col, config.data.target_transform
    )
    metadata['target_transform'] = target_meta

    # Get original features (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    metadata['n_original_features'] = len(feature_cols)

    # Start with original features
    X = df[feature_cols].copy()

    # Create lag features
    lag_cols = config.features.lag_features
    if lag_cols:
        # If target is in lag_cols, use the raw target column
        if target_col in lag_cols or 'CO2e' in lag_cols:
            lag_target_df = create_lag_features(
                df[[target_col]],
                [target_col],
                config.features.lag_orders
            )
            X = pd.concat([X, lag_target_df], axis=1)
            metadata['lag_features'].extend(lag_target_df.columns.tolist())

        # Lag other features
        other_lag_cols = [c for c in lag_cols if c in X.columns]
        if other_lag_cols:
            lag_df = create_lag_features(
                X[other_lag_cols],
                other_lag_cols,
                config.features.lag_orders
            )
            X = pd.concat([X, lag_df], axis=1)
            metadata['lag_features'].extend(lag_df.columns.tolist())

    # Create rolling features (if enabled)
    if getattr(config.features, 'include_rolling_features', False):
        rolling_cols = getattr(config.features, 'rolling_columns', ['CO2e'])
        rolling_windows = getattr(config.features, 'rolling_windows', [4, 8])
        rolling_funcs = getattr(config.features, 'rolling_functions', ['mean', 'std'])

        # Use target column for rolling features
        roll_df_target = df[[target_col]].copy()
        roll_df_target.columns = ['CO2e']  # Normalize name

        rolling_df = create_rolling_features(
            roll_df_target,
            ['CO2e'],
            windows=rolling_windows,
            functions=rolling_funcs
        )
        # Shift rolling features by 1 to avoid data leakage
        rolling_df = rolling_df.shift(1)
        X = pd.concat([X, rolling_df], axis=1)
        metadata['rolling_features'] = rolling_df.columns.tolist()
        logger.info(f"Created {len(rolling_df.columns)} rolling features (shifted by 1 to avoid leakage)")

    # Create rate-of-change features (if enabled)
    if getattr(config.features, 'include_roc_features', False):
        roc_cols = getattr(config.features, 'roc_columns', ['TEC', 'GDP'])
        # Also add target column rate-of-change
        roc_all_cols = roc_cols + [target_col]
        roc_df = create_rate_of_change_features(
            df[list(set(roc_all_cols) & set(df.columns))],
            list(set(roc_all_cols) & set(df.columns)),
            log_transform=True
        )
        # Shift by 1 to avoid leakage for target-related features
        for col in roc_df.columns:
            if target_col in col:
                roc_df[col] = roc_df[col].shift(1)
        X = pd.concat([X, roc_df], axis=1)
        metadata['roc_features'] = roc_df.columns.tolist()
        logger.info(f"Created {len(roc_df.columns)} rate-of-change features")

    # Create intensity features (if enabled)
    if getattr(config.features, 'include_intensity_features', False):
        intensity_denominators = getattr(config.features, 'intensity_denominators', ['Population', 'TEC'])
        intensity_df = create_intensity_features(
            df,
            target_col=target_col,
            denominator_cols=intensity_denominators
        )
        X = pd.concat([X, intensity_df], axis=1)
        metadata['intensity_features'] = intensity_df.columns.tolist()
        logger.info(f"Created {len(intensity_df.columns)} intensity features")

    # Create weather features (if enabled)
    if getattr(config.features, 'include_weather_features', False):
        temp_col = getattr(config.features, 'temperature_column', 'Air_Temp')
        base_temp = getattr(config.features, 'hdd_base_temp', 18.0)
        weather_df = create_weather_features(df, temp_col=temp_col, base_temp=base_temp)
        X = pd.concat([X, weather_df], axis=1)
        metadata['weather_features'] = weather_df.columns.tolist()
        logger.info(f"Created {len(weather_df.columns)} weather features")

    # Create seasonality features
    season_df = create_seasonality_features(df, config.features.seasonality_type)
    X = pd.concat([X, season_df], axis=1)
    metadata['seasonality_features'] = season_df.columns.tolist()

    # Create shock features
    shock_df = create_shock_features(df, config.features)
    X = pd.concat([X, shock_df], axis=1)
    metadata['shock_features'] = shock_df.columns.tolist()

    metadata['feature_columns'] = X.columns.tolist()
    metadata['n_total_features'] = len(X.columns)

    logger.info(f"Feature engineering complete. Total features: {metadata['n_total_features']}")

    return X, y, metadata


def create_feature_dictionary(
    X: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create a feature dictionary describing all features.

    Args:
        X: Feature DataFrame
        metadata: Feature metadata
        output_path: Path to save dictionary (optional)

    Returns:
        Feature dictionary DataFrame
    """
    feature_dict = []

    for col in X.columns:
        entry = {
            'feature': col,
            'dtype': str(X[col].dtype),
            'n_unique': X[col].nunique(),
            'min': X[col].min() if X[col].dtype in [np.float64, np.int64] else None,
            'max': X[col].max() if X[col].dtype in [np.float64, np.int64] else None,
            'mean': X[col].mean() if X[col].dtype in [np.float64, np.int64] else None,
            'missing_pct': (X[col].isnull().sum() / len(X)) * 100
        }

        # Categorize feature type
        if col in metadata.get('lag_features', []):
            entry['category'] = 'lag'
        elif col in metadata.get('rolling_features', []):
            entry['category'] = 'rolling'
        elif col in metadata.get('seasonality_features', []):
            entry['category'] = 'seasonality'
        elif col in metadata.get('shock_features', []):
            entry['category'] = 'shock'
        else:
            entry['category'] = 'original'

        feature_dict.append(entry)

    df_dict = pd.DataFrame(feature_dict)

    if output_path:
        df_dict.to_csv(output_path, index=False)

    return df_dict
