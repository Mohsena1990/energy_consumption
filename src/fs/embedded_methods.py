"""
Embedded feature selection methods.

Embedded methods perform feature selection as part of the
model training process (e.g., L1 regularization, tree importance).
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from ..core.logging_utils import get_logger


def lasso_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = None,
    cv: int = 5,
    max_iter: int = 10000,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    LASSO-based feature selection (L1 regularization).

    LASSO shrinks some coefficients to exactly zero,
    effectively performing feature selection.

    Args:
        X: Feature DataFrame
        y: Target Series
        alpha: Regularization strength (None = use CV)
        cv: Cross-validation folds for alpha selection
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, coefficients DataFrame)
    """
    logger = get_logger()
    logger.info("Running LASSO feature selection...")

    # Scale features for fair regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use cross-validation to find optimal alpha if not specified
    if alpha is None:
        lasso_cv = LassoCV(cv=cv, random_state=seed, max_iter=max_iter)
        lasso_cv.fit(X_scaled, y)
        alpha = lasso_cv.alpha_
        logger.info(f"LASSO optimal alpha via CV: {alpha:.6f}")

    # Fit LASSO with selected alpha
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=seed)
    lasso.fit(X_scaled, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lasso.coef_,
        'abs_coefficient': np.abs(lasso.coef_),
        'selected': np.abs(lasso.coef_) > 1e-10
    }).sort_values('abs_coefficient', ascending=False)

    selected = scores_df[scores_df['selected']]['feature'].tolist()

    logger.info(f"LASSO: Selected {len(selected)} features (alpha={alpha:.6f})")
    return selected, scores_df


def elasticnet_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = None,
    l1_ratio: float = 0.5,
    cv: int = 5,
    max_iter: int = 10000,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    ElasticNet feature selection (L1 + L2 regularization).

    Combines LASSO (L1) and Ridge (L2) penalties.
    l1_ratio controls the mix (1 = pure LASSO, 0 = pure Ridge).

    Args:
        X: Feature DataFrame
        y: Target Series
        alpha: Regularization strength (None = use CV)
        l1_ratio: Balance between L1 and L2 (0-1)
        cv: Cross-validation folds
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, coefficients DataFrame)
    """
    logger = get_logger()
    logger.info(f"Running ElasticNet feature selection (l1_ratio={l1_ratio})...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use CV for alpha if not specified
    if alpha is None:
        en_cv = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, random_state=seed, max_iter=max_iter)
        en_cv.fit(X_scaled, y)
        alpha = en_cv.alpha_
        logger.info(f"ElasticNet optimal alpha: {alpha:.6f}")

    # Fit ElasticNet
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=seed)
    en.fit(X_scaled, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': en.coef_,
        'abs_coefficient': np.abs(en.coef_),
        'selected': np.abs(en.coef_) > 1e-10
    }).sort_values('abs_coefficient', ascending=False)

    selected = scores_df[scores_df['selected']]['feature'].tolist()

    logger.info(f"ElasticNet: Selected {len(selected)} features")
    return selected, scores_df


def gradient_boosting_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    n_estimators: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Gradient Boosting feature importance selection.

    Uses LightGBM (if available) or sklearn GradientBoosting
    to compute feature importances.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        n_estimators: Number of boosting iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    logger = get_logger()
    logger.info("Running Gradient Boosting feature selection...")

    # Try LightGBM first, fallback to sklearn
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            verbosity=-1,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X, y)
        importances = model.feature_importances_
        logger.info("Using LightGBM for feature importance")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed
        )
        model.fit(X, y)
        importances = model.feature_importances_
        logger.info("Using sklearn GradientBoosting for feature importance")

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Normalize
    total_importance = scores_df['importance'].sum()
    if total_importance > 0:
        scores_df['importance_normalized'] = scores_df['importance'] / total_importance
    else:
        scores_df['importance_normalized'] = 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"Gradient Boosting: Selected {len(selected)} features")
    return selected, scores_df


def random_forest_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    n_estimators: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Random Forest feature importance selection.

    Uses mean decrease in impurity (MDI) importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        n_estimators: Number of trees
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    from sklearn.ensemble import RandomForestRegressor

    logger = get_logger()
    logger.info("Running Random Forest feature selection...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X, y)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Normalize
    total = scores_df['importance'].sum()
    scores_df['importance_normalized'] = scores_df['importance'] / total if total > 0 else 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"Random Forest: Selected {len(selected)} features")
    return selected, scores_df


def catboost_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    threshold: float = None,
    iterations: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    CatBoost feature importance selection.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Number of top features to select
        threshold: Alternative - importance threshold
        iterations: Number of boosting iterations
        seed: Random seed

    Returns:
        Tuple of (selected feature names, importance DataFrame)
    """
    logger = get_logger()
    logger.info("Running CatBoost feature selection...")

    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.1,
            depth=5,
            verbose=False,
            random_seed=seed,
            allow_writing_files=False
        )
        model.fit(X, y)
        importances = model.feature_importances_
    except ImportError:
        logger.warning("CatBoost not available. Using Gradient Boosting fallback.")
        return gradient_boosting_selection(X, y, top_k, threshold, n_estimators=iterations, seed=seed)

    scores_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Normalize
    total = scores_df['importance'].sum()
    scores_df['importance_normalized'] = scores_df['importance'] / total if total > 0 else 0

    # Select features
    if threshold is not None:
        selected = scores_df[scores_df['importance_normalized'] >= threshold]['feature'].tolist()
    else:
        selected = scores_df.head(top_k)['feature'].tolist()

    logger.info(f"CatBoost: Selected {len(selected)} features")
    return selected, scores_df


def run_all_embedded_methods(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    cv: int = 5,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Run all embedded feature selection methods.

    Args:
        X: Feature DataFrame
        y: Target Series
        top_k: Target number of features
        cv: Cross-validation folds
        seed: Random seed

    Returns:
        Dictionary with results from all embedded methods
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Running ALL embedded feature selection methods...")

    results = {}

    # LASSO
    try:
        lasso_selected, lasso_scores = lasso_selection(X, y, cv=cv, seed=seed)
        results['lasso'] = {
            'method': 'embedded',
            'type': 'lasso',
            'selected_features': lasso_selected,
            'n_selected': len(lasso_selected),
            'scores': lasso_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"LASSO failed: {e}")

    # ElasticNet
    try:
        en_selected, en_scores = elasticnet_selection(X, y, cv=cv, seed=seed)
        results['elasticnet'] = {
            'method': 'embedded',
            'type': 'elasticnet',
            'selected_features': en_selected,
            'n_selected': len(en_selected),
            'scores': en_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"ElasticNet failed: {e}")

    # Gradient Boosting
    try:
        gb_selected, gb_scores = gradient_boosting_selection(X, y, top_k=top_k, seed=seed)
        results['gradient_boosting'] = {
            'method': 'embedded',
            'type': 'gradient_boosting',
            'selected_features': gb_selected,
            'n_selected': len(gb_selected),
            'scores': gb_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Gradient Boosting failed: {e}")

    # Random Forest
    try:
        rf_selected, rf_scores = random_forest_selection(X, y, top_k=top_k, seed=seed)
        results['random_forest'] = {
            'method': 'embedded',
            'type': 'random_forest',
            'selected_features': rf_selected,
            'n_selected': len(rf_selected),
            'scores': rf_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Random Forest failed: {e}")

    # CatBoost
    try:
        cb_selected, cb_scores = catboost_selection(X, y, top_k=top_k, seed=seed)
        results['catboost'] = {
            'method': 'embedded',
            'type': 'catboost',
            'selected_features': cb_selected,
            'n_selected': len(cb_selected),
            'scores': cb_scores.to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"CatBoost failed: {e}")

    logger.info(f"Embedded methods complete: {len(results)} methods run")
    return results
