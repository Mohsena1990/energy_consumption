"""
Consensus feature selection: combining multiple FS methods.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..splits.walk_forward import CVPlan
from .linear import fs_linear
from .nonlinear import fs_nonlinear
from .filter_methods import run_all_filter_methods
from .wrapper_methods import run_all_wrapper_methods
from .embedded_methods import run_all_embedded_methods


def vote_based_selection(
    method_selections: Dict[str, List[str]],
    all_features: List[str],
    min_votes: int = 2
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on voting across methods.

    Args:
        method_selections: Dict mapping method name to selected features
        all_features: List of all possible features
        min_votes: Minimum number of methods that must select a feature

    Returns:
        Tuple of (selected features, vote counts)
    """
    logger = get_logger()

    # Count votes for each feature
    vote_counts = Counter()
    for method, features in method_selections.items():
        for f in features:
            vote_counts[f] += 1

    # Create vote summary
    vote_df = pd.DataFrame([
        {
            'feature': f,
            'vote_count': vote_counts.get(f, 0),
            'selected_by': [m for m, feats in method_selections.items() if f in feats]
        }
        for f in all_features
    ])
    vote_df['selected_by'] = vote_df['selected_by'].apply(lambda x: ', '.join(x))
    vote_df = vote_df.sort_values('vote_count', ascending=False)

    # Select features with minimum votes
    selected = vote_df[vote_df['vote_count'] >= min_votes]['feature'].tolist()

    logger.info(f"Vote-based selection: {len(selected)} features with >= {min_votes} votes")

    return selected, vote_df


def stability_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    method: str = 'lightgbm',
    stability_threshold: float = 0.6,
    top_k: int = 10,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features based on importance stability across CV folds.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        method: Model to use for importance ('lightgbm' or 'rf')
        stability_threshold: Fraction of folds where feature must be in top-K
        top_k: Number of top features to consider per fold
        seed: Random seed

    Returns:
        Tuple of (selected features, stability scores)
    """
    logger = get_logger()

    # Track top-K selections per fold
    top_k_counts = {col: 0 for col in X.columns}

    # Get unique folds
    seen_folds = set()
    unique_folds = []
    n_folds = 0

    for fold in cv_plan.folds:
        fold_key = (tuple(fold.train_indices), tuple(fold.test_indices))
        if fold_key not in seen_folds:
            seen_folds.add(fold_key)
            unique_folds.append(fold)
            n_folds += 1

    for fold in unique_folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]

        if method == 'lightgbm':
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.1,
                    verbosity=-1, random_state=seed
                )
                model.fit(X_train, y_train)
                importances = model.feature_importances_
            except ImportError:
                method = 'rf'

        if method == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=seed, n_jobs=-1
            )
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        # Get top-K features for this fold
        importance_order = np.argsort(importances)[::-1]
        top_k_features = [X.columns[i] for i in importance_order[:top_k]]

        for f in top_k_features:
            top_k_counts[f] += 1

    # Calculate stability (fraction of folds in top-K)
    stability_df = pd.DataFrame([
        {
            'feature': col,
            'top_k_count': count,
            'stability': count / n_folds if n_folds > 0 else 0
        }
        for col, count in top_k_counts.items()
    ]).sort_values('stability', ascending=False)

    # Select features above stability threshold
    selected = stability_df[stability_df['stability'] >= stability_threshold]['feature'].tolist()

    logger.info(f"Stability-based selection: {len(selected)} features with stability >= {stability_threshold}")

    return selected, stability_df


def fs_consensus(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config,
    linear_results: Dict[str, Any] = None,
    nonlinear_results: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Consensus feature selection combining linear and nonlinear methods.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration
        linear_results: Pre-computed linear FS results (optional)
        nonlinear_results: Pre-computed nonlinear FS results (optional)

    Returns:
        Dictionary with selected features and scores
    """
    logger = get_logger()
    logger.info("Running consensus feature selection (FS_consensus)...")

    results = {
        'method': 'consensus',
        'steps': []
    }

    # Run or use provided linear FS
    if linear_results is None:
        linear_results = fs_linear(X, y, cv_plan, config)

    # Run or use provided nonlinear FS
    if nonlinear_results is None:
        nonlinear_results = fs_nonlinear(X, y, cv_plan, config)

    # Collect all method selections
    method_selections = {}

    # From linear methods
    for step in linear_results.get('steps', []):
        method_name = f"linear_{step['name']}"
        method_selections[method_name] = step.get('selected', [])

    # From nonlinear methods
    for step in nonlinear_results.get('steps', []):
        method_name = f"nonlinear_{step['name']}"
        method_selections[method_name] = step.get('selected', [])

    # Step 1: Vote-based selection
    vote_selected, vote_df = vote_based_selection(
        method_selections,
        list(X.columns),
        min_votes=config.fs.vote_threshold
    )
    results['steps'].append({
        'name': 'vote_selection',
        'selected': vote_selected,
        'n_selected': len(vote_selected),
        'votes': vote_df.to_dict(orient='records')
    })

    # Step 2: Stability-based selection
    stability_selected, stability_df = stability_based_selection(
        X, y, cv_plan,
        method=config.fs.evaluator_model,
        stability_threshold=config.fs.stability_threshold,
        seed=config.seed
    )
    results['steps'].append({
        'name': 'stability_selection',
        'selected': stability_selected,
        'n_selected': len(stability_selected),
        'stability': stability_df.to_dict(orient='records')
    })

    # Final consensus: intersection of vote and stability
    final_selected = list(set(vote_selected) & set(stability_selected))

    # If intersection is too small, use union with higher vote threshold
    if len(final_selected) < 3:
        logger.warning("Intersection too small, using features with >=3 votes or high stability")
        high_vote = vote_df[vote_df['vote_count'] >= 3]['feature'].tolist()
        high_stability = stability_df[stability_df['stability'] >= 0.7]['feature'].tolist()
        final_selected = list(set(high_vote) | set(high_stability))

    # Preserve original order
    final_selected = [f for f in X.columns if f in final_selected]

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)
    results['linear_results'] = linear_results
    results['nonlinear_results'] = nonlinear_results

    logger.info(f"FS_consensus final: {len(final_selected)} features selected")

    return results


def fs_hybrid(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config,
    filter_results: Dict[str, Dict] = None,
    wrapper_results: Dict[str, Dict] = None,
    embedded_results: Dict[str, Dict] = None
) -> Dict[str, Any]:
    """
    Hybrid feature selection combining filter, wrapper, and embedded methods.

    This approach combines:
    - Filter methods: Fast statistical pre-screening
    - Wrapper methods: Model-based iterative selection
    - Embedded methods: Feature importance during training

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration
        filter_results: Pre-computed filter FS results (optional)
        wrapper_results: Pre-computed wrapper FS results (optional)
        embedded_results: Pre-computed embedded FS results (optional)

    Returns:
        Dictionary with selected features and scores
    """
    logger = get_logger()
    logger.info("Running hybrid feature selection (filter+wrapper+embedded)...")

    results = {
        'method': 'hybrid',
        'steps': []
    }

    # Get configuration
    top_k = getattr(config.fs, 'top_k_features', 10)
    min_votes = getattr(config.fs, 'hybrid_min_votes', 2)
    min_features = getattr(config.fs, 'min_features', 3)

    # Run filter methods if not provided
    if filter_results is None:
        logger.info("Running filter methods...")
        filter_results = run_all_filter_methods(X, y, top_k=top_k, seed=config.seed)

    # Run wrapper methods if not provided
    if wrapper_results is None:
        logger.info("Running wrapper methods...")
        wrapper_results = run_all_wrapper_methods(X, y, n_features=top_k, cv=5, seed=config.seed)

    # Run embedded methods if not provided
    if embedded_results is None:
        logger.info("Running embedded methods...")
        embedded_results = run_all_embedded_methods(X, y, top_k=top_k, cv=5, seed=config.seed)

    # Store results
    results['filter_results'] = filter_results
    results['wrapper_results'] = wrapper_results
    results['embedded_results'] = embedded_results

    # Collect all method selections
    method_selections = {}

    # From filter methods
    for method_name, method_result in filter_results.items():
        method_selections[f"filter_{method_name}"] = method_result.get('selected_features', [])

    # From wrapper methods
    for method_name, method_result in wrapper_results.items():
        method_selections[f"wrapper_{method_name}"] = method_result.get('selected_features', [])

    # From embedded methods
    for method_name, method_result in embedded_results.items():
        method_selections[f"embedded_{method_name}"] = method_result.get('selected_features', [])

    # Step 1: Count votes across all methods
    vote_counts = Counter()
    for method, features in method_selections.items():
        for f in features:
            vote_counts[f] += 1

    vote_df = pd.DataFrame([
        {
            'feature': f,
            'vote_count': vote_counts.get(f, 0),
            'method_count': len(method_selections),
            'vote_pct': vote_counts.get(f, 0) / len(method_selections) if method_selections else 0
        }
        for f in X.columns
    ]).sort_values('vote_count', ascending=False)

    results['steps'].append({
        'name': 'vote_aggregation',
        'votes': vote_df.to_dict(orient='records'),
        'n_methods': len(method_selections)
    })

    # Step 2: Category-wise selection (ensure diversity)
    filter_union = set()
    for method_result in filter_results.values():
        filter_union.update(method_result.get('selected_features', []))

    wrapper_union = set()
    for method_result in wrapper_results.values():
        wrapper_union.update(method_result.get('selected_features', []))

    embedded_union = set()
    for method_result in embedded_results.values():
        embedded_union.update(method_result.get('selected_features', []))

    # Features selected by at least one method from each category
    multi_category = filter_union & wrapper_union & embedded_union
    logger.info(f"Features selected by all three categories: {len(multi_category)}")

    # Features with at least min_votes across all methods
    high_vote = set(vote_df[vote_df['vote_count'] >= min_votes]['feature'].tolist())
    logger.info(f"Features with >= {min_votes} votes: {len(high_vote)}")

    # Final selection: union of multi-category and high-vote features
    final_selected = list(multi_category | high_vote)

    # Ensure minimum features (fallback to top-voted)
    if len(final_selected) < min_features:
        logger.info(f"Hybrid: Only {len(final_selected)} features, adding top-voted features")
        remaining = vote_df[~vote_df['feature'].isin(final_selected)].head(min_features - len(final_selected))
        final_selected.extend(remaining['feature'].tolist())

    # Preserve original order
    final_selected = [f for f in X.columns if f in final_selected]

    results['steps'].append({
        'name': 'hybrid_selection',
        'multi_category_count': len(multi_category),
        'high_vote_count': len(high_vote),
        'filter_union_count': len(filter_union),
        'wrapper_union_count': len(wrapper_union),
        'embedded_union_count': len(embedded_union)
    })

    results['selected_features'] = final_selected
    results['n_selected'] = len(final_selected)

    logger.info(f"FS_hybrid final: {len(final_selected)} features selected")

    return results


def run_all_fs_options(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Dict[str, Dict[str, Any]]:
    """
    Run all feature selection options including hybrid approach.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Dictionary with results for each FS option
    """
    logger = get_logger()
    logger.info("Running all feature selection options...")

    # Clean data for FS
    X_clean = X.select_dtypes(include=[np.number]).dropna()
    y_clean = y.loc[X_clean.index].dropna()
    common_idx = X_clean.index.intersection(y_clean.index)
    X_clean = X_clean.loc[common_idx]
    y_clean = y_clean.loc[common_idx]

    top_k = getattr(config.fs, 'top_k_features', 10)
    run_hybrid = getattr(config.fs, 'run_hybrid_fs', True)

    results = {}

    # FS_linear
    logger.info("=" * 50)
    results['fs_linear'] = fs_linear(X_clean, y_clean, cv_plan, config)

    # FS_nonlinear
    logger.info("=" * 50)
    results['fs_nonlinear'] = fs_nonlinear(X_clean, y_clean, cv_plan, config)

    # FS_consensus (uses results from linear and nonlinear)
    logger.info("=" * 50)
    results['fs_consensus'] = fs_consensus(
        X_clean, y_clean, cv_plan, config,
        linear_results=results['fs_linear'],
        nonlinear_results=results['fs_nonlinear']
    )

    # Run filter, wrapper, embedded methods for additional options and hybrid
    if run_hybrid:
        logger.info("=" * 50)
        logger.info("Running filter methods...")
        filter_results = run_all_filter_methods(X_clean, y_clean, top_k=top_k, seed=config.seed)

        logger.info("=" * 50)
        logger.info("Running wrapper methods...")
        wrapper_results = run_all_wrapper_methods(X_clean, y_clean, n_features=top_k, cv=5, seed=config.seed)

        logger.info("=" * 50)
        logger.info("Running embedded methods...")
        embedded_results = run_all_embedded_methods(X_clean, y_clean, top_k=top_k, cv=5, seed=config.seed)

        # Add individual method results as FS options
        results['fs_filter'] = {
            'method': 'filter',
            'steps': list(filter_results.values()),
            'selected_features': list(set().union(*[
                set(r.get('selected_features', [])) for r in filter_results.values()
            ])),
            'n_selected': len(set().union(*[
                set(r.get('selected_features', [])) for r in filter_results.values()
            ])),
            'sub_methods': filter_results
        }

        results['fs_wrapper'] = {
            'method': 'wrapper',
            'steps': list(wrapper_results.values()),
            'selected_features': list(set().union(*[
                set(r.get('selected_features', [])) for r in wrapper_results.values()
            ])),
            'n_selected': len(set().union(*[
                set(r.get('selected_features', [])) for r in wrapper_results.values()
            ])),
            'sub_methods': wrapper_results
        }

        results['fs_embedded'] = {
            'method': 'embedded',
            'steps': list(embedded_results.values()),
            'selected_features': list(set().union(*[
                set(r.get('selected_features', [])) for r in embedded_results.values()
            ])),
            'n_selected': len(set().union(*[
                set(r.get('selected_features', [])) for r in embedded_results.values()
            ])),
            'sub_methods': embedded_results
        }

        # FS_hybrid (filter + wrapper + embedded)
        logger.info("=" * 50)
        results['fs_hybrid'] = fs_hybrid(
            X_clean, y_clean, cv_plan, config,
            filter_results=filter_results,
            wrapper_results=wrapper_results,
            embedded_results=embedded_results
        )

    logger.info("=" * 50)
    logger.info(f"Total FS options: {len(results)}")
    for fs_name, fs_result in results.items():
        logger.info(f"  {fs_name}: {fs_result.get('n_selected', 0)} features")

    return results
