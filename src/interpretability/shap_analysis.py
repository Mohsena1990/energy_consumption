"""
SHAP-based interpretability for CO2 forecasting models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from ..core.logging_utils import get_logger


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = 'auto'
) -> Tuple[np.ndarray, Any]:
    """
    Compute SHAP values for a model.

    Args:
        model: Trained model
        X: Feature data
        model_type: Type of explainer ('tree', 'kernel', 'auto')

    Returns:
        Tuple of (SHAP values array, explainer)
    """
    logger = get_logger()
    import shap

    if model_type == 'auto':
        # Detect model type
        model_name = type(model).__name__.lower()
        if any(t in model_name for t in ['forest', 'tree', 'lgbm', 'lightgbm', 'catboost', 'xgb']):
            model_type = 'tree'
        elif 'ridge' in model_name or 'linear' in model_name:
            model_type = 'linear'
        else:
            model_type = 'kernel'

    logger.info(f"Computing SHAP values using {model_type} explainer")

    if model_type == 'tree':
        # For tree-based models
        if hasattr(model, 'model'):
            explainer = shap.TreeExplainer(model.model)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    elif model_type == 'linear':
        # For linear models
        if hasattr(model, 'model'):
            explainer = shap.LinearExplainer(model.model, X)
        else:
            explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    else:
        # For other models, use Kernel SHAP (slower)
        background = shap.sample(X, min(100, len(X)))
        if hasattr(model, 'predict'):
            predict_fn = model.predict
        else:
            predict_fn = model
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X)

    return shap_values, explainer


def get_feature_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances
    """
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    # Normalize to sum to 1
    importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()

    return importance_df


def analyze_regime_shap(
    model,
    X: pd.DataFrame,
    regime_column: str = None,
    regime_periods: Dict[str, Tuple[str, str]] = None,
    model_type: str = 'auto'
) -> Dict[str, pd.DataFrame]:
    """
    Compare SHAP importances across different regimes (e.g., pre/post COVID).

    Args:
        model: Trained model
        X: Feature DataFrame with datetime index
        regime_column: Column indicating regime (if available)
        regime_periods: Dict mapping regime name to (start_date, end_date) tuples
        model_type: Type of SHAP explainer

    Returns:
        Dictionary mapping regime to importance DataFrame
    """
    logger = get_logger()

    results = {}

    if regime_periods is None:
        # Default regime periods
        regime_periods = {
            'pre_covid': (None, '2020-01-01'),
            'covid': ('2020-01-01', '2022-01-01'),
            'post_covid': ('2022-01-01', None)
        }

    feature_names = list(X.columns)

    for regime_name, (start, end) in regime_periods.items():
        # Filter data by regime
        mask = pd.Series(True, index=X.index)

        if start is not None:
            mask &= (X.index >= start)
        if end is not None:
            mask &= (X.index < end)

        X_regime = X[mask]

        if len(X_regime) < 5:
            logger.warning(f"Regime '{regime_name}' has only {len(X_regime)} samples, skipping")
            continue

        # Compute SHAP values
        try:
            shap_values, _ = compute_shap_values(model, X_regime, model_type)
            importance_df = get_feature_importance_from_shap(shap_values, feature_names)
            importance_df['regime'] = regime_name
            importance_df['n_samples'] = len(X_regime)
            results[regime_name] = importance_df

            logger.info(f"Regime '{regime_name}': Top feature = {importance_df.iloc[0]['feature']}")

        except Exception as e:
            logger.warning(f"SHAP analysis failed for regime '{regime_name}': {e}")

    return results


def analyze_seasonal_shap(
    model,
    X: pd.DataFrame,
    model_type: str = 'auto'
) -> pd.DataFrame:
    """
    Analyze SHAP importances by season/quarter.

    Args:
        model: Trained model
        X: Feature DataFrame with datetime index
        model_type: Type of SHAP explainer

    Returns:
        DataFrame with seasonal importance analysis
    """
    logger = get_logger()

    feature_names = list(X.columns)
    seasonal_results = []

    for quarter in [1, 2, 3, 4]:
        X_quarter = X[X.index.quarter == quarter]

        if len(X_quarter) < 3:
            continue

        try:
            shap_values, _ = compute_shap_values(model, X_quarter, model_type)
            importance = get_feature_importance_from_shap(shap_values, feature_names)

            for _, row in importance.iterrows():
                seasonal_results.append({
                    'quarter': f'Q{quarter}',
                    'feature': row['feature'],
                    'importance': row['importance'],
                    'importance_normalized': row['importance_normalized']
                })

        except Exception as e:
            logger.warning(f"SHAP analysis failed for Q{quarter}: {e}")

    return pd.DataFrame(seasonal_results)


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute permutation feature importance.

    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target values
        n_repeats: Number of permutation repeats
        seed: Random seed

    Returns:
        DataFrame with permutation importances
    """
    from sklearn.inspection import permutation_importance

    logger = get_logger()

    # Get predictions
    if hasattr(model, 'predict'):
        predict_fn = model.predict
    else:
        predict_fn = model

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring='neg_mean_absolute_error'
    )

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    # Flip sign (neg_mae -> positive importance means worse performance when shuffled)
    importance_df['importance_mean'] = -importance_df['importance_mean']

    logger.info(f"Top feature by permutation importance: {importance_df.iloc[0]['feature']}")

    return importance_df


def get_ridge_coefficients(model) -> pd.DataFrame:
    """
    Get coefficients from Ridge model.

    Args:
        model: Ridge model (or wrapper)

    Returns:
        DataFrame with coefficients
    """
    if hasattr(model, 'model'):
        sklearn_model = model.model
        feature_names = model.feature_names
    else:
        sklearn_model = model
        feature_names = [f'feature_{i}' for i in range(len(model.coef_))]

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': sklearn_model.coef_
    })

    coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

    return coef_df


def generate_interpretation_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive interpretation report for a model.

    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        model_name: Name of the model
        output_dir: Directory to save results

    Returns:
        Interpretation report dictionary
    """
    logger = get_logger()
    logger.info(f"Generating interpretation report for {model_name}")

    report = {
        'model': model_name,
        'n_features': len(X.columns),
        'n_samples': len(X)
    }

    # Get feature importance
    if model_name.lower() == 'ridge':
        # Use coefficients for Ridge
        coef_df = get_ridge_coefficients(model)
        report['coefficients'] = coef_df.to_dict(orient='records')

        # Also compute permutation importance
        perm_df = compute_permutation_importance(model, X, y)
        report['permutation_importance'] = perm_df.to_dict(orient='records')

    elif hasattr(model, 'supports_shap') and model.supports_shap:
        # Use SHAP for tree-based models
        try:
            shap_values, _ = compute_shap_values(model, X, 'tree')
            feature_names = list(X.columns)

            importance_df = get_feature_importance_from_shap(shap_values, feature_names)
            report['shap_importance'] = importance_df.to_dict(orient='records')

            # Regime analysis
            regime_results = analyze_regime_shap(model, X, model_type='tree')
            report['regime_analysis'] = {k: v.to_dict(orient='records') for k, v in regime_results.items()}

            # Seasonal analysis
            seasonal_df = analyze_seasonal_shap(model, X, model_type='tree')
            report['seasonal_analysis'] = seasonal_df.to_dict(orient='records')

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            # Fallback to permutation importance
            perm_df = compute_permutation_importance(model, X, y)
            report['permutation_importance'] = perm_df.to_dict(orient='records')

    else:
        # Fallback to permutation importance
        perm_df = compute_permutation_importance(model, X, y)
        report['permutation_importance'] = perm_df.to_dict(orient='records')

    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_dir / f'{model_name}_interpretation.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

    return report
