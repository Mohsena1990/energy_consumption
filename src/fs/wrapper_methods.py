"""
Wrapper-based feature selection methods.

Wrapper methods use a machine learning model to evaluate
feature subsets by training and scoring the model.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from ..core.logging_utils import get_logger


def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features_to_select: int = 10,
    estimator_type: str = 'ridge',
    step: int = 1,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Recursive Feature Elimination (RFE).

    RFE recursively removes the least important features
    based on the model's feature weights.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_features_to_select: Number of features to keep
        estimator_type: 'ridge' or 'random_forest'
        step: Number of features to remove at each iteration
        seed: Random seed

    Returns:
        Tuple of (selected feature names, ranking DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running RFE feature selection (target: {n_features_to_select} features)...")

    # Select estimator
    if estimator_type == 'ridge':
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                          random_state=seed, n_jobs=-1)

    # Run RFE
    rfe = RFE(estimator, n_features_to_select=min(n_features_to_select, len(X.columns)),
              step=step)
    rfe.fit(X, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'ranking': rfe.ranking_,
        'selected': rfe.support_
    }).sort_values('ranking')

    selected = scores_df[scores_df['selected']]['feature'].tolist()

    logger.info(f"RFE: Selected {len(selected)} features using {estimator_type}")
    return selected, scores_df


def sequential_forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features_to_select: int = 10,
    cv: int = 5,
    estimator_type: str = 'ridge',
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Sequential Forward Selection (SFS).

    Starts with empty set and adds features one by one
    that maximize model performance.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_features_to_select: Number of features to select
        cv: Cross-validation folds
        estimator_type: 'ridge' or 'random_forest'
        scoring: Scoring metric for CV
        seed: Random seed

    Returns:
        Tuple of (selected feature names, selection DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running Sequential Forward Selection (target: {n_features_to_select} features)...")

    # Select estimator
    if estimator_type == 'ridge':
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                          random_state=seed, n_jobs=-1)

    # Run SFS
    n_select = min(n_features_to_select, len(X.columns))
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_select,
        direction='forward',
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    sfs.fit(X, y)

    selected = list(X.columns[sfs.get_support()])

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'selected': sfs.get_support(),
        'selection_order': [selected.index(f) + 1 if f in selected else 0
                           for f in X.columns]
    })

    logger.info(f"SFS: Selected {len(selected)} features using {estimator_type}")
    return selected, scores_df


def sequential_backward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features_to_select: int = 10,
    cv: int = 5,
    estimator_type: str = 'ridge',
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Sequential Backward Selection (SBS).

    Starts with all features and removes features one by one
    that have the least impact on model performance.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_features_to_select: Number of features to keep
        cv: Cross-validation folds
        estimator_type: 'ridge' or 'random_forest'
        scoring: Scoring metric for CV
        seed: Random seed

    Returns:
        Tuple of (selected feature names, selection DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running Sequential Backward Selection (target: {n_features_to_select} features)...")

    # Select estimator
    if estimator_type == 'ridge':
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5,
                                          random_state=seed, n_jobs=-1)

    # Run SBS
    n_select = min(n_features_to_select, len(X.columns))
    sbs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_select,
        direction='backward',
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    sbs.fit(X, y)

    selected = list(X.columns[sbs.get_support()])

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'selected': sbs.get_support()
    })

    logger.info(f"SBS: Selected {len(selected)} features using {estimator_type}")
    return selected, scores_df


def exhaustive_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 1,
    max_features: int = 5,
    cv: int = 5,
    scoring: str = 'neg_mean_absolute_error',
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Exhaustive feature selection (for small feature sets).

    Tests all possible combinations within the size range.
    Warning: Computationally expensive for large feature sets!

    Args:
        X: Feature DataFrame (should be small, <15 features)
        y: Target Series
        min_features: Minimum subset size
        max_features: Maximum subset size
        cv: Cross-validation folds
        scoring: Scoring metric
        seed: Random seed

    Returns:
        Tuple of (best feature subset, results DataFrame)
    """
    from itertools import combinations

    logger = get_logger()

    if len(X.columns) > 15:
        logger.warning("Exhaustive search not recommended for >15 features. Using top 15.")
        # Use correlation to pre-filter
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        X = X[correlations.head(15).index]

    logger.info(f"Running exhaustive feature selection ({len(X.columns)} features)...")

    estimator = Ridge(alpha=1.0)
    results = []

    for size in range(min_features, min(max_features + 1, len(X.columns) + 1)):
        for combo in combinations(X.columns, size):
            X_subset = X[list(combo)]
            scores = cross_val_score(estimator, X_subset, y, cv=cv, scoring=scoring)
            results.append({
                'features': combo,
                'n_features': size,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            })

    results_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)

    # Best subset
    best_row = results_df.iloc[0]
    selected = list(best_row['features'])

    logger.info(f"Exhaustive: Selected {len(selected)} features (score: {best_row['mean_score']:.4f})")
    return selected, results_df


def run_all_wrapper_methods(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 10,
    cv: int = 5,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all wrapper-based feature selection methods.

    Args:
        X: Feature DataFrame
        y: Target Series
        n_features: Target number of features
        cv: Cross-validation folds
        seed: Random seed

    Returns:
        Dictionary with results from all wrapper methods
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Running ALL wrapper-based feature selection methods...")

    results = {}

    # RFE with Ridge
    try:
        rfe_selected, rfe_scores = rfe_selection(X, y, n_features_to_select=n_features,
                                                  estimator_type='ridge', seed=seed)
        results['rfe_ridge'] = {
            'method': 'wrapper',
            'type': 'rfe',
            'estimator': 'ridge',
            'selected_features': rfe_selected,
            'n_selected': len(rfe_selected),
            'scores': rfe_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"RFE (Ridge) failed: {e}")

    # RFE with Random Forest
    try:
        rfe_rf_selected, rfe_rf_scores = rfe_selection(X, y, n_features_to_select=n_features,
                                                        estimator_type='random_forest', seed=seed)
        results['rfe_rf'] = {
            'method': 'wrapper',
            'type': 'rfe',
            'estimator': 'random_forest',
            'selected_features': rfe_rf_selected,
            'n_selected': len(rfe_rf_selected),
            'scores': rfe_rf_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"RFE (RF) failed: {e}")

    # Sequential Forward Selection
    try:
        sfs_selected, sfs_scores = sequential_forward_selection(X, y, n_features_to_select=n_features,
                                                                 cv=cv, seed=seed)
        results['sfs'] = {
            'method': 'wrapper',
            'type': 'sequential_forward',
            'selected_features': sfs_selected,
            'n_selected': len(sfs_selected),
            'scores': sfs_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"SFS failed: {e}")

    # Sequential Backward Selection
    try:
        sbs_selected, sbs_scores = sequential_backward_selection(X, y, n_features_to_select=n_features,
                                                                   cv=cv, seed=seed)
        results['sbs'] = {
            'method': 'wrapper',
            'type': 'sequential_backward',
            'selected_features': sbs_selected,
            'n_selected': len(sbs_selected),
            'scores': sbs_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"SBS failed: {e}")

    logger.info(f"Wrapper methods complete: {len(results)} methods run")
    return results
