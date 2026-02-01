"""
Filter-based feature selection methods.

Filter methods select features based on statistical measures
without using a machine learning model.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr

from ..core.logging_utils import get_logger


def mutual_information_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features using Mutual Information regression.

    Mutual information measures the dependency between the feature
    and the target variable. Higher MI = more informative feature.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative selection by MI threshold
        seed: Random seed for reproducibility

    Returns:
        Tuple of (selected feature names, scores DataFrame)
    """
    logger = get_logger()
    logger.info("Running Mutual Information feature selection...")

    # Compute MI scores
    mi_scores = mutual_info_regression(X, y, random_state=seed)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores,
        'mi_normalized': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
    }).sort_values('mi_score', ascending=False)

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['mi_score'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"Mutual Information: Selected {len(selected)} features (max MI: {mi_scores.max():.4f})")
    return selected, scores_df


def f_test_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    p_value_threshold: float = 0.05
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features using F-test regression.

    F-test measures the linear relationship between each feature
    and the target. Features with low p-values are significant.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        p_value_threshold: P-value threshold for significance

    Returns:
        Tuple of (selected feature names, scores DataFrame)
    """
    logger = get_logger()
    logger.info("Running F-test feature selection...")

    # Compute F-scores and p-values
    f_scores, p_values = f_regression(X, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'p_value': p_values,
        'significant': p_values <= p_value_threshold
    }).sort_values('f_score', ascending=False)

    # Select by p-value threshold
    selected = scores_df[scores_df['p_value'] <= p_value_threshold]['feature'].tolist()

    # Ensure at least top_k if threshold is too strict
    if len(selected) < top_k:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"F-test: Selected {len(selected)} features (significant at p<{p_value_threshold})")
    return selected, scores_df


def correlation_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.3,
    top_k: int = None,
    method: str = 'pearson'
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on correlation with target.

    Args:
        X: Feature DataFrame
        y: Target Series
        correlation_threshold: Minimum absolute correlation
        top_k: Alternative - select top K by correlation
        method: 'pearson' or 'spearman'

    Returns:
        Tuple of (selected feature names, scores DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running {method} correlation-based feature selection...")

    correlations = []
    corr_func = pearsonr if method == 'pearson' else spearmanr

    for col in X.columns:
        try:
            valid_idx = ~(X[col].isna() | y.isna())
            if valid_idx.sum() > 2:
                corr, p_val = corr_func(X[col][valid_idx], y[valid_idx])
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'p_value': p_val
                })
        except Exception as e:
            logger.warning(f"Correlation failed for {col}: {e}")
            correlations.append({
                'feature': col,
                'correlation': 0.0,
                'abs_correlation': 0.0,
                'p_value': 1.0
            })

    scores_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['abs_correlation'] >= correlation_threshold]['feature'].tolist()

    logger.info(f"Correlation-based: Selected {len(selected)} features (threshold={correlation_threshold})")
    return selected, scores_df


def variance_threshold_selection(
    X: pd.DataFrame,
    threshold: float = 0.01
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on variance threshold.

    Removes features with variance below the threshold (low variability).

    Args:
        X: Feature DataFrame
        threshold: Minimum variance threshold (after scaling)

    Returns:
        Tuple of (selected feature names, variance DataFrame)
    """
    logger = get_logger()
    logger.info("Running variance threshold feature selection...")

    # Scale features to [0,1] for fair variance comparison
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    variances = X_scaled.var()

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'variance': variances.values,
        'above_threshold': variances.values >= threshold
    }).sort_values('variance', ascending=False)

    selected = scores_df[scores_df['above_threshold']]['feature'].tolist()

    logger.info(f"Variance threshold: Selected {len(selected)} features (threshold={threshold})")
    return selected, scores_df


def run_all_filter_methods(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all filter-based feature selection methods.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of features to select per method
        seed: Random seed

    Returns:
        Dictionary with results from all filter methods
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Running ALL filter-based feature selection methods...")

    results = {}

    # Mutual Information
    try:
        mi_selected, mi_scores = mutual_information_selection(X, y, top_k=top_k, seed=seed)
        results['mutual_info'] = {
            'method': 'filter',
            'type': 'mutual_information',
            'selected_features': mi_selected,
            'n_selected': len(mi_selected),
            'scores': mi_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Mutual Information failed: {e}")

    # F-test
    try:
        f_selected, f_scores = f_test_selection(X, y, top_k=top_k)
        results['f_test'] = {
            'method': 'filter',
            'type': 'f_test',
            'selected_features': f_selected,
            'n_selected': len(f_selected),
            'scores': f_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"F-test failed: {e}")

    # Pearson Correlation
    try:
        corr_selected, corr_scores = correlation_based_selection(X, y, top_k=top_k, method='pearson')
        results['correlation'] = {
            'method': 'filter',
            'type': 'pearson_correlation',
            'selected_features': corr_selected,
            'n_selected': len(corr_selected),
            'scores': corr_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Correlation failed: {e}")

    # Variance Threshold
    try:
        var_selected, var_scores = variance_threshold_selection(X, threshold=0.01)
        results['variance'] = {
            'method': 'filter',
            'type': 'variance_threshold',
            'selected_features': var_selected,
            'n_selected': len(var_selected),
            'scores': var_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Variance threshold failed: {e}")

    logger.info(f"Filter methods complete: {len(results)} methods run")
    return results
