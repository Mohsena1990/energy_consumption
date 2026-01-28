"""
Data quality audit and cleaning for CO2 forecasting framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from ..core.logging_utils import get_logger


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing value statistics per column
    """
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    report = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values,
        'dtype': df.dtypes.values
    })

    return report.sort_values('missing_pct', ascending=False).reset_index(drop=True)


def check_constant_columns(df: pd.DataFrame, threshold: float = 0.99) -> List[str]:
    """
    Find columns with nearly constant values.

    Args:
        df: Input DataFrame
        threshold: Percentage of same values to be considered constant

    Returns:
        List of constant column names
    """
    constant_cols = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            mode_pct = df[col].value_counts(normalize=True).max()
            if mode_pct >= threshold:
                constant_cols.append(col)
        else:
            # For non-numeric, check if only one unique value
            if df[col].nunique() <= 1:
                constant_cols.append(col)

    return constant_cols


def check_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Flag outliers in numeric columns.

    Args:
        df: Input DataFrame
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)

    Returns:
        DataFrame with outlier flags
    """
    outlier_report = []

    for col in df.select_dtypes(include=[np.number]).columns:
        values = df[col].dropna()

        if len(values) == 0:
            continue

        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = (values < lower) | (values > upper)
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            if std > 0:
                z_scores = np.abs((values - mean) / std)
                outliers = z_scores > threshold
            else:
                outliers = pd.Series([False] * len(values))
        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = outliers.sum()
        if n_outliers > 0:
            outlier_report.append({
                'column': col,
                'n_outliers': n_outliers,
                'pct_outliers': (n_outliers / len(values)) * 100,
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std()
            })

    return pd.DataFrame(outlier_report)


def check_temporal_gaps(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for gaps in temporal data.

    Args:
        df: DataFrame with datetime index

    Returns:
        Dictionary with gap information
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {'error': 'Index is not DatetimeIndex'}

    # Expected quarterly frequency
    expected_freq = pd.DateOffset(months=3)

    gaps = []
    for i in range(1, len(df.index)):
        diff = (df.index[i] - df.index[i-1])
        if diff > pd.Timedelta(days=100):  # More than ~3 months
            gaps.append({
                'from': df.index[i-1],
                'to': df.index[i],
                'gap_days': diff.days
            })

    return {
        'n_gaps': len(gaps),
        'gaps': gaps,
        'date_range': {
            'start': df.index.min(),
            'end': df.index.max(),
            'n_periods': len(df.index)
        }
    }


def generate_quality_report(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.

    Args:
        df: Input DataFrame
        output_dir: Directory to save report (optional)

    Returns:
        Dictionary with quality report
    """
    logger = get_logger()
    logger.info("Generating data quality report...")

    report = {
        'summary': {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
                'end': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
            }
        }
    }

    # Missing values
    missing_report = check_missing_values(df)
    report['missing_values'] = missing_report.to_dict(orient='records')

    # Constant columns
    constant_cols = check_constant_columns(df)
    report['constant_columns'] = constant_cols

    # Outliers
    outlier_report = check_outliers(df)
    report['outliers'] = outlier_report.to_dict(orient='records') if len(outlier_report) > 0 else []

    # Temporal gaps
    temporal_gaps = check_temporal_gaps(df)
    report['temporal_gaps'] = temporal_gaps

    # Column statistics
    numeric_stats = df.describe().to_dict()
    report['numeric_statistics'] = numeric_stats

    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save missing values report
        missing_report.to_csv(output_dir / 'missing_values.csv', index=False)

        # Save outlier report
        if len(outlier_report) > 0:
            outlier_report.to_csv(output_dir / 'outliers.csv', index=False)

        # Save full report as JSON
        import json
        with open(output_dir / 'data_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Quality report saved to: {output_dir}")

    return report


def clean_data(
    df: pd.DataFrame,
    drop_constant: bool = True,
    interpolate_missing: bool = True,
    interpolate_method: str = 'linear',
    max_missing_pct: float = 50.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean data based on quality checks.

    Args:
        df: Input DataFrame
        drop_constant: Whether to drop constant columns
        interpolate_missing: Whether to interpolate missing values
        interpolate_method: Interpolation method
        max_missing_pct: Maximum percentage of missing values to keep column

    Returns:
        Tuple of (cleaned DataFrame, cleaning log)
    """
    logger = get_logger()
    df_clean = df.copy()
    cleaning_log = {
        'dropped_columns': [],
        'interpolated_columns': [],
        'remaining_missing': {}
    }

    # Drop constant columns
    if drop_constant:
        constant_cols = check_constant_columns(df_clean)
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
            df_clean = df_clean.drop(columns=constant_cols)
            cleaning_log['dropped_columns'].extend(constant_cols)

    # Drop columns with too many missing values
    missing_report = check_missing_values(df_clean)
    high_missing = missing_report[missing_report['missing_pct'] > max_missing_pct]['column'].tolist()
    if high_missing:
        logger.info(f"Dropping {len(high_missing)} columns with >{max_missing_pct}% missing: {high_missing}")
        df_clean = df_clean.drop(columns=high_missing)
        cleaning_log['dropped_columns'].extend(high_missing)

    # Interpolate missing values
    if interpolate_missing:
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in [np.float64, np.int64]:
                    df_clean[col] = df_clean[col].interpolate(method=interpolate_method)
                    cleaning_log['interpolated_columns'].append(col)

    # Check remaining missing
    remaining_missing = df_clean.isnull().sum()
    cleaning_log['remaining_missing'] = remaining_missing[remaining_missing > 0].to_dict()

    logger.info(f"Cleaning complete. Shape: {df_clean.shape}")

    return df_clean, cleaning_log
