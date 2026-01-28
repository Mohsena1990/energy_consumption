"""
Feature selection evaluation using SHAP.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import spearmanr

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..core.utils import calculate_weighted_mae
from ..splits.walk_forward import CVPlan, generate_cv_folds


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = 'tree'
) -> np.ndarray:
    """
    Compute SHAP values for a model.

    Args:
        model: Trained model
        X: Feature data
        model_type: Type of model ('tree' for tree-based, 'kernel' for others)

    Returns:
        SHAP values array
    """
    import shap

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)

    return shap_values


def evaluate_fs_option_with_shap(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    cv_plan: CVPlan,
    config: Config,
    fs_name: str
) -> Dict[str, Any]:
    """
    Evaluate a feature selection option using SHAP-based metrics.

    Args:
        X: Full feature DataFrame
        y: Target Series
        selected_features: Features selected by this FS method
        cv_plan: CV plan
        config: Configuration
        fs_name: Name of FS method

    Returns:
        Dictionary with evaluation metrics
    """
    logger = get_logger()
    logger.info(f"Evaluating FS option: {fs_name} ({len(selected_features)} features)")

    # Use only selected features
    X_selected = X[selected_features].copy()

    # Initialize evaluator model
    if config.fs.evaluator_model == 'lightgbm':
        try:
            import lightgbm as lgb
            model_class = lgb.LGBMRegressor
            model_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'verbosity': -1,
                'random_state': config.seed
            }
            model_type = 'tree'
        except ImportError:
            logger.warning("LightGBM not available, using CatBoost")
            config.fs.evaluator_model = 'catboost'

    if config.fs.evaluator_model == 'catboost':
        from catboost import CatBoostRegressor
        model_class = CatBoostRegressor
        model_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'verbose': False,
            'random_seed': config.seed
        }
        model_type = 'tree'

    # Track metrics across folds
    fold_errors = {h: [] for h in config.splits.horizons}
    fold_shap_importances = []
    fold_shap_top_k = []
    all_predictions = []

    import shap

    # Get unique folds for primary horizon (h=1) for SHAP stability
    h1_folds = [f for f in cv_plan.folds if f.horizon == 1]

    for fold in cv_plan.folds:
        # Get train/test data
        X_train = X_selected.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]
        X_test = X_selected.iloc[fold.test_indices]
        y_test = y.iloc[fold.test_indices]

        # Skip if not enough data
        if len(X_train) < 10 or len(X_test) < 1:
            continue

        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate error
        mae = np.mean(np.abs(y_test.values - y_pred))
        fold_errors[fold.horizon].append(mae)

        # Store predictions
        for idx, (true_val, pred_val) in enumerate(zip(y_test.values, y_pred)):
            all_predictions.append({
                'fold_id': fold.fold_id,
                'horizon': fold.horizon,
                'date': y_test.index[idx],
                'actual': true_val,
                'predicted': pred_val,
                'error': true_val - pred_val
            })

        # Compute SHAP values (only for h=1 folds to avoid redundancy)
        if fold.horizon == 1:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                # Mean absolute SHAP importance per feature
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                shap_importance = dict(zip(selected_features, mean_abs_shap))
                fold_shap_importances.append(shap_importance)

                # Top-K features by SHAP
                top_k = config.fs.top_k_features
                sorted_features = sorted(shap_importance.items(), key=lambda x: -x[1])
                top_k_features = set([f[0] for f in sorted_features[:top_k]])
                fold_shap_top_k.append(top_k_features)
            except Exception as e:
                logger.warning(f"SHAP computation failed for fold {fold.fold_id}: {e}")

    # ========== Compute Evaluation Metrics ==========

    results = {
        'fs_name': fs_name,
        'n_features': len(selected_features),
        'features': selected_features
    }

    # C1: Accuracy - Weighted MAE across horizons
    horizon_maes = {}
    for h in config.splits.horizons:
        if fold_errors[h]:
            horizon_maes[h] = np.mean(fold_errors[h])
        else:
            horizon_maes[h] = np.inf

    results['mae_h1'] = horizon_maes.get(1, np.inf)
    results['mae_h2'] = horizon_maes.get(2, np.inf)
    results['mae_h4'] = horizon_maes.get(4, np.inf)
    results['weighted_mae'] = calculate_weighted_mae(horizon_maes, config.splits.horizon_weights)

    # C2: Stability - std of error across folds + worst fold
    all_errors = []
    for h in config.splits.horizons:
        all_errors.extend(fold_errors[h])

    if all_errors:
        results['error_std'] = np.std(all_errors)
        results['worst_fold_error'] = np.max(all_errors)
        results['stability_score'] = 1 / (1 + results['error_std'])  # Higher is better
    else:
        results['error_std'] = np.inf
        results['worst_fold_error'] = np.inf
        results['stability_score'] = 0

    # C3: SHAP concentration - share of total |SHAP| by top-K features
    if fold_shap_importances:
        # Average SHAP importance across folds
        avg_shap = {}
        for feat in selected_features:
            values = [imp.get(feat, 0) for imp in fold_shap_importances]
            avg_shap[feat] = np.mean(values)

        total_shap = sum(avg_shap.values())
        if total_shap > 0:
            sorted_shap = sorted(avg_shap.items(), key=lambda x: -x[1])
            top_k = min(config.fs.top_k_features, len(sorted_shap))
            top_k_sum = sum([s[1] for s in sorted_shap[:top_k]])
            results['shap_concentration'] = top_k_sum / total_shap
        else:
            results['shap_concentration'] = 0

        results['avg_shap_importance'] = avg_shap
    else:
        results['shap_concentration'] = 0
        results['avg_shap_importance'] = {}

    # C4: SHAP stability - rank correlation + Jaccard of top-K
    if len(fold_shap_importances) >= 2:
        # Rank correlation between consecutive folds
        rank_correlations = []
        for i in range(len(fold_shap_importances) - 1):
            imp1 = fold_shap_importances[i]
            imp2 = fold_shap_importances[i + 1]
            # Align features
            common_features = list(set(imp1.keys()) & set(imp2.keys()))
            if len(common_features) >= 3:
                vals1 = [imp1[f] for f in common_features]
                vals2 = [imp2[f] for f in common_features]
                corr, _ = spearmanr(vals1, vals2)
                if not np.isnan(corr):
                    rank_correlations.append(corr)

        results['shap_rank_correlation'] = np.mean(rank_correlations) if rank_correlations else 0

        # Jaccard similarity of top-K features
        jaccard_scores = []
        for i in range(len(fold_shap_top_k) - 1):
            jac = jaccard_similarity(fold_shap_top_k[i], fold_shap_top_k[i + 1])
            jaccard_scores.append(jac)

        results['shap_top_k_jaccard'] = np.mean(jaccard_scores) if jaccard_scores else 0
        results['shap_stability'] = (results['shap_rank_correlation'] + results['shap_top_k_jaccard']) / 2
    else:
        results['shap_rank_correlation'] = 0
        results['shap_top_k_jaccard'] = 0
        results['shap_stability'] = 0

    # C5: Parsimony - number of features (lower is better)
    results['parsimony'] = len(selected_features)
    # Normalize parsimony score (fewer features = higher score)
    max_features = len(X.columns)
    results['parsimony_score'] = 1 - (len(selected_features) / max_features)

    # Store predictions
    results['predictions'] = pd.DataFrame(all_predictions)

    logger.info(f"  Weighted MAE: {results['weighted_mae']:.4f}")
    logger.info(f"  Stability: {results['stability_score']:.4f}")
    logger.info(f"  SHAP concentration: {results['shap_concentration']:.4f}")
    logger.info(f"  SHAP stability: {results['shap_stability']:.4f}")
    logger.info(f"  Parsimony: {results['parsimony']} features")

    return results


def create_fs_evaluation_matrix(
    fs_results: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Create evaluation matrix for all FS options.

    Args:
        fs_results: Results from run_all_fs_options
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration

    Returns:
        Tuple of (evaluation matrix DataFrame, detailed results dict)
    """
    logger = get_logger()
    logger.info("Creating FS evaluation matrix...")

    # Clean data
    X_clean = X.select_dtypes(include=[np.number]).dropna()
    y_clean = y.loc[X_clean.index].dropna()
    common_idx = X_clean.index.intersection(y_clean.index)
    X_clean = X_clean.loc[common_idx]
    y_clean = y_clean.loc[common_idx]

    detailed_results = {}
    matrix_rows = []

    for fs_name, fs_result in fs_results.items():
        selected_features = fs_result.get('selected_features', [])

        if not selected_features:
            logger.warning(f"No features selected for {fs_name}, skipping evaluation")
            continue

        # Filter to features that exist in X_clean
        selected_features = [f for f in selected_features if f in X_clean.columns]

        if not selected_features:
            logger.warning(f"No valid features for {fs_name} after filtering")
            continue

        # Evaluate
        eval_results = evaluate_fs_option_with_shap(
            X_clean, y_clean, selected_features, cv_plan, config, fs_name
        )

        detailed_results[fs_name] = eval_results

        # Add to matrix
        matrix_rows.append({
            'fs_option': fs_name,
            'n_features': eval_results['n_features'],
            'C1_accuracy': eval_results['weighted_mae'],
            'C2_stability': eval_results['stability_score'],
            'C3_shap_concentration': eval_results['shap_concentration'],
            'C4_shap_stability': eval_results['shap_stability'],
            'C5_parsimony': eval_results['parsimony_score'],
            'mae_h1': eval_results['mae_h1'],
            'mae_h2': eval_results['mae_h2'],
            'mae_h4': eval_results['mae_h4'],
            'error_std': eval_results['error_std']
        })

    matrix_df = pd.DataFrame(matrix_rows)
    logger.info(f"Evaluation matrix created with {len(matrix_df)} FS options")

    return matrix_df, detailed_results
