"""
Linear feature selection methods: VIF + Ridge/ElasticNet stability.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan, generate_cv_folds
from .vif import filter_by_vif


def ridge_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    alpha: float = 1.0,
    threshold: float = 0.01,
    stability_threshold: float = 0.6
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on Ridge coefficient stability across CV folds.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        alpha: Ridge regularization strength
        threshold: Minimum absolute coefficient to be considered important
        stability_threshold: Minimum fraction of folds where feature must be important

    Returns:
        Tuple of (selected features, stability scores)
    """
    logger = get_logger()

    feature_importance = {col: [] for col in X.columns}
    fold_coefficients = []

    # Get unique folds (not per horizon, just unique train/test splits)
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        # Store coefficients
        coef_dict = dict(zip(X.columns, np.abs(model.coef_)))
        fold_coefficients.append(coef_dict)

        # Track importance per feature
        for col, coef in coef_dict.items():
            feature_importance[col].append(int(coef >= threshold))

    # Calculate stability score (fraction of folds where feature is important)
    stability_scores = []
    for col in X.columns:
        scores = feature_importance[col]
        stability = np.mean(scores) if scores else 0
        avg_coef = np.mean([fc[col] for fc in fold_coefficients])
        std_coef = np.std([fc[col] for fc in fold_coefficients])
        stability_scores.append({
            'feature': col,
            'stability': stability,
            'avg_coefficient': avg_coef,
            'std_coefficient': std_coef,
            'n_folds_important': sum(scores),
            'n_folds_total': len(scores)
        })

    stability_df = pd.DataFrame(stability_scores).sort_values('stability', ascending=False)

    # Select features above stability threshold
    selected = stability_df[stability_df['stability'] >= stability_threshold]['feature'].tolist()

    logger.info(f"Ridge stability: Selected {len(selected)} features (stability >= {stability_threshold})")

    return selected, stability_df


def elasticnet_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    alpha: float = 0.5,
    l1_ratio: float = 0.5,
    stability_threshold: float = 0.6
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on ElasticNet coefficient stability across CV folds.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        alpha: Regularization strength
        l1_ratio: L1/L2 ratio (1.0 = Lasso, 0.0 = Ridge)
        stability_threshold: Minimum fraction of folds where feature must have non-zero coefficient

    Returns:
        Tuple of (selected features, stability scores)
    """
    logger = get_logger()

    feature_nonzero = {col: [] for col in X.columns}
    fold_coefficients = []

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_train_scaled, y_train)

        # Store coefficients
        coef_dict = dict(zip(X.columns, model.coef_))
        fold_coefficients.append(coef_dict)

        # Track non-zero coefficients
        for col, coef in coef_dict.items():
            feature_nonzero[col].append(int(np.abs(coef) > 1e-10))

    # Calculate stability score
    stability_scores = []
    for col in X.columns:
        nonzero = feature_nonzero[col]
        stability = np.mean(nonzero) if nonzero else 0
        avg_coef = np.mean([fc[col] for fc in fold_coefficients])
        std_coef = np.std([fc[col] for fc in fold_coefficients])
        stability_scores.append({
            'feature': col,
            'stability': stability,
            'avg_coefficient': avg_coef,
            'std_coefficient': std_coef,
            'n_folds_nonzero': sum(nonzero),
            'n_folds_total': len(nonzero)
        })

    stability_df = pd.DataFrame(stability_scores).sort_values('stability', ascending=False)

    # Select features above stability threshold
    selected = stability_df[stability_df['stability'] >= stability_threshold]['feature'].tolist()

    logger.info(f"ElasticNet stability: Selected {len(selected)} features (stability >= {stability_threshold})")

    return selected, stability_df


def fs_linear(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    Full linear feature selection pipeline: VIF filter + Ridge/ElasticNet stability.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with selected features and scores
    """
    logger = get_logger()
    logger.info("Running linear feature selection (FS_linear)...")

    results = {
        'method': 'linear',
        'steps': []
    }

    # Step 1: VIF filter
    vif_selected, vif_report = filter_by_vif(X, threshold=config.fs.vif_threshold)
    results['steps'].append({
        'name': 'vif_filter',
        'selected': vif_selected,
        'n_selected': len(vif_selected),
        'report': vif_report.to_dict(orient='records')
    })

    X_vif = X[vif_selected]

    # Step 2: Ridge stability
    ridge_selected, ridge_scores = ridge_stability_selection(
        X_vif, y, cv_plan,
        stability_threshold=config.fs.stability_threshold
    )
    results['steps'].append({
        'name': 'ridge_stability',
        'selected': ridge_selected,
        'n_selected': len(ridge_selected),
        'scores': ridge_scores.to_dict(orient='records')
    })

    # Step 3: ElasticNet stability
    enet_selected, enet_scores = elasticnet_stability_selection(
        X_vif, y, cv_plan,
        stability_threshold=config.fs.stability_threshold
    )
    results['steps'].append({
        'name': 'elasticnet_stability',
        'selected': enet_selected,
        'n_selected': len(enet_selected),
        'scores': enet_scores.to_dict(orient='records')
    })

    # Final selection: union of Ridge and ElasticNet stable features
    final_selected = list(set(ridge_selected) | set(enet_selected))
    final_selected = [f for f in X_vif.columns if f in final_selected]  # Preserve order

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_linear final: {len(final_selected)} features selected")

    return results
