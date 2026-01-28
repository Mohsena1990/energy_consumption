"""
VIF (Variance Inflation Factor) for multicollinearity filtering.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..core.logging_utils import get_logger


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for all features.

    Args:
        X: Feature DataFrame (numeric only, no NaN)

    Returns:
        DataFrame with VIF scores
    """
    logger = get_logger()

    # Ensure numeric only
    X_numeric = X.select_dtypes(include=[np.number]).dropna()

    if len(X_numeric.columns) == 0:
        return pd.DataFrame(columns=['feature', 'vif'])

    # Add constant for VIF calculation
    X_const = X_numeric.copy()
    X_const['const'] = 1

    vif_data = []
    for i, col in enumerate(X_numeric.columns):
        try:
            vif = variance_inflation_factor(X_const.values, i)
            vif_data.append({'feature': col, 'vif': vif})
        except Exception as e:
            logger.warning(f"Could not calculate VIF for {col}: {e}")
            vif_data.append({'feature': col, 'vif': np.inf})

    return pd.DataFrame(vif_data).sort_values('vif', ascending=False)


def filter_by_vif(
    X: pd.DataFrame,
    threshold: float = 10.0,
    iterative: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filter features by VIF threshold.

    Args:
        X: Feature DataFrame
        threshold: VIF threshold (default 10.0)
        iterative: If True, iteratively remove highest VIF until all below threshold

    Returns:
        Tuple of (selected features, VIF report)
    """
    logger = get_logger()

    X_filtered = X.select_dtypes(include=[np.number]).dropna().copy()
    selected_features = list(X_filtered.columns)
    all_vif_scores = []

    if not iterative:
        # One-pass: calculate VIF and remove features above threshold
        vif_df = calculate_vif(X_filtered)
        all_vif_scores.append(vif_df)
        selected_features = vif_df[vif_df['vif'] <= threshold]['feature'].tolist()
    else:
        # Iterative: remove highest VIF feature until all below threshold
        iteration = 0
        while True:
            vif_df = calculate_vif(X_filtered[selected_features])
            vif_df['iteration'] = iteration
            all_vif_scores.append(vif_df)

            max_vif = vif_df['vif'].max()
            if max_vif <= threshold or len(selected_features) <= 1:
                break

            # Remove feature with highest VIF
            worst_feature = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
            selected_features.remove(worst_feature)
            logger.info(f"Iteration {iteration}: Removed '{worst_feature}' (VIF={max_vif:.2f})")
            iteration += 1

    vif_report = pd.concat(all_vif_scores, ignore_index=True)
    logger.info(f"VIF filter: {len(X_filtered.columns)} -> {len(selected_features)} features")

    return selected_features, vif_report
