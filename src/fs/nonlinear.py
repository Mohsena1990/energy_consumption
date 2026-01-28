"""
Nonlinear feature selection methods: RF importance, Boruta, LightGBM/CatBoost importance.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan


def random_forest_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    n_estimators: int = 100,
    max_depth: int = 10,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on Random Forest feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        top_k: Select top K features (if None, use threshold)
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    importance_scores = {col: [] for col in X.columns}

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

        # Fit Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Store importances
        for col, imp in zip(X.columns, model.feature_importances_):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['mean_importance'] >= threshold]['feature'].tolist()

    logger.info(f"RF importance: Selected {len(selected)} features")

    return selected, scores_df


def lightgbm_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    params: Optional[Dict] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on LightGBM feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        params: LightGBM parameters
        top_k: Select top K features
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed, skipping LightGBM importance")
        return [], pd.DataFrame()

    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': seed
        }

    importance_scores = {col: [] for col in X.columns}

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

        # Fit LightGBM
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # Store importances (gain-based)
        importances = model.feature_importances_
        for col, imp in zip(X.columns, importances):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Normalize to 0-1 range
    max_imp = scores_df['mean_importance'].max()
    if max_imp > 0:
        scores_df['normalized_importance'] = scores_df['mean_importance'] / max_imp
    else:
        scores_df['normalized_importance'] = 0

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['normalized_importance'] >= threshold]['feature'].tolist()

    logger.info(f"LightGBM importance: Selected {len(selected)} features")

    return selected, scores_df


def catboost_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    params: Optional[Dict] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on CatBoost feature importance.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        params: CatBoost parameters
        top_k: Select top K features
        threshold: Minimum importance threshold
        seed: Random seed

    Returns:
        Tuple of (selected features, importance scores)
    """
    logger = get_logger()

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("CatBoost not installed, skipping CatBoost importance")
        return [], pd.DataFrame()

    if params is None:
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False,
            'random_seed': seed
        }

    importance_scores = {col: [] for col in X.columns}

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

        # Fit CatBoost
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)

        # Store importances
        importances = model.feature_importances_
        for col, imp in zip(X.columns, importances):
            importance_scores[col].append(imp)

    # Aggregate importance scores
    scores_df = pd.DataFrame([
        {
            'feature': col,
            'mean_importance': np.mean(scores),
            'std_importance': np.std(scores),
            'min_importance': np.min(scores),
            'max_importance': np.max(scores)
        }
        for col, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    # Normalize
    max_imp = scores_df['mean_importance'].max()
    if max_imp > 0:
        scores_df['normalized_importance'] = scores_df['mean_importance'] / max_imp
    else:
        scores_df['normalized_importance'] = 0

    # Select features
    if top_k is not None:
        selected = scores_df.head(top_k)['feature'].tolist()
    else:
        selected = scores_df[scores_df['normalized_importance'] >= threshold]['feature'].tolist()

    logger.info(f"CatBoost importance: Selected {len(selected)} features")

    return selected, scores_df


def boruta_selection(
    X: pd.DataFrame,
    y: pd.Series,
    max_iter: int = 100,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features using Boruta algorithm.

    Args:
        X: Feature DataFrame
        y: Target Series
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Tuple of (selected features, Boruta results)
    """
    logger = get_logger()

    try:
        from boruta import BorutaPy
    except ImportError:
        logger.warning("Boruta not installed, using RF importance as fallback")
        # Fallback to simple RF importance
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
        rf.fit(X, y)
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        selected = importances.head(len(X.columns) // 2)['feature'].tolist()
        return selected, importances

    # Fit Boruta
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=max_iter, random_state=seed)

    # Handle NaN values
    X_clean = X.fillna(X.mean())
    boruta.fit(X_clean.values, y.values)

    # Get results
    results = pd.DataFrame({
        'feature': X.columns,
        'ranking': boruta.ranking_,
        'selected': boruta.support_,
        'weak': boruta.support_weak_
    }).sort_values('ranking')

    selected = results[results['selected']]['feature'].tolist()

    # Also include weak features if few selected
    if len(selected) < 3:
        weak = results[results['weak']]['feature'].tolist()
        selected = list(set(selected + weak))

    logger.info(f"Boruta: Selected {len(selected)} features")

    return selected, results


def fs_nonlinear(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Any]:
    """
    Full nonlinear feature selection pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with selected features and scores
    """
    logger = get_logger()
    logger.info("Running nonlinear feature selection (FS_nonlinear)...")

    results = {
        'method': 'nonlinear',
        'steps': []
    }

    # Step 1: Random Forest importance
    rf_selected, rf_scores = random_forest_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'random_forest',
        'selected': rf_selected,
        'n_selected': len(rf_selected),
        'scores': rf_scores.to_dict(orient='records')
    })

    # Step 2: LightGBM importance
    lgb_selected, lgb_scores = lightgbm_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'lightgbm',
        'selected': lgb_selected,
        'n_selected': len(lgb_selected),
        'scores': lgb_scores.to_dict(orient='records') if len(lgb_scores) > 0 else []
    })

    # Step 3: CatBoost importance
    cat_selected, cat_scores = catboost_importance(
        X, y, cv_plan, seed=config.seed
    )
    results['steps'].append({
        'name': 'catboost',
        'selected': cat_selected,
        'n_selected': len(cat_selected),
        'scores': cat_scores.to_dict(orient='records') if len(cat_scores) > 0 else []
    })

    # Final selection: union of tree-based methods
    all_selected = set(rf_selected) | set(lgb_selected) | set(cat_selected)
    final_selected = [f for f in X.columns if f in all_selected]

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_nonlinear final: {len(final_selected)} features selected")

    return results
