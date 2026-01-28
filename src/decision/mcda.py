"""
Multi-Criteria Decision Analysis (MCDA) methods: VIKOR and TOPSIS.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

from ..core.logging_utils import get_logger


def normalize_matrix(
    matrix: np.ndarray,
    criteria_types: List[str]
) -> np.ndarray:
    """
    Normalize decision matrix using vector normalization.

    Args:
        matrix: Decision matrix (alternatives x criteria)
        criteria_types: List of 'benefit' or 'cost' for each criterion

    Returns:
        Normalized matrix
    """
    norm_matrix = np.zeros_like(matrix, dtype=float)

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        denom = np.sqrt(np.sum(col ** 2))
        if denom > 0:
            norm_matrix[:, j] = col / denom
        else:
            norm_matrix[:, j] = 0

    return norm_matrix


def pareto_filter(
    df: pd.DataFrame,
    criteria: List[str],
    criteria_types: Dict[str, str]
) -> pd.DataFrame:
    """
    Filter to Pareto-optimal (non-dominated) alternatives.

    Args:
        df: DataFrame with alternatives
        criteria: List of criteria column names
        criteria_types: Dict mapping criterion to 'benefit' or 'cost'

    Returns:
        DataFrame with only Pareto-optimal alternatives
    """
    logger = get_logger()

    n = len(df)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            dominates = True
            strictly_better_in_one = False

            for c in criteria:
                val_i = df.iloc[i][c]
                val_j = df.iloc[j][c]

                if criteria_types.get(c, 'benefit') == 'benefit':
                    # Higher is better
                    if val_j < val_i:
                        dominates = False
                        break
                    if val_j > val_i:
                        strictly_better_in_one = True
                else:
                    # Lower is better (cost)
                    if val_j > val_i:
                        dominates = False
                        break
                    if val_j < val_i:
                        strictly_better_in_one = True

            if dominates and strictly_better_in_one:
                is_dominated[i] = True
                break

    pareto_df = df[~np.array(is_dominated)].copy()
    logger.info(f"Pareto filter: {n} -> {len(pareto_df)} alternatives")

    return pareto_df


def topsis(
    df: pd.DataFrame,
    criteria: List[str],
    weights: Dict[str, float],
    criteria_types: Dict[str, str]
) -> pd.DataFrame:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

    Args:
        df: DataFrame with alternatives and criteria columns
        criteria: List of criteria column names
        weights: Dict mapping criterion to weight (should sum to 1)
        criteria_types: Dict mapping criterion to 'benefit' or 'cost'

    Returns:
        DataFrame with TOPSIS scores and rankings
    """
    logger = get_logger()

    # Extract decision matrix
    matrix = df[criteria].values.astype(float)
    n_alternatives, n_criteria = matrix.shape

    # Normalize weights
    w = np.array([weights.get(c, 1/n_criteria) for c in criteria])
    w = w / w.sum()

    # Step 1: Normalize matrix
    norm_matrix = normalize_matrix(matrix, list(criteria_types.values()))

    # Step 2: Weighted normalized matrix
    weighted_matrix = norm_matrix * w

    # Step 3: Ideal and anti-ideal solutions
    ideal = np.zeros(n_criteria)
    anti_ideal = np.zeros(n_criteria)

    for j, c in enumerate(criteria):
        col = weighted_matrix[:, j]
        if criteria_types.get(c, 'benefit') == 'benefit':
            ideal[j] = np.max(col)
            anti_ideal[j] = np.min(col)
        else:
            ideal[j] = np.min(col)
            anti_ideal[j] = np.max(col)

    # Step 4: Distance to ideal and anti-ideal
    d_plus = np.sqrt(np.sum((weighted_matrix - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_matrix - anti_ideal) ** 2, axis=1))

    # Step 5: Relative closeness (higher is better)
    closeness = d_minus / (d_plus + d_minus + 1e-10)

    # Create results
    result_df = df.copy()
    result_df['topsis_d_plus'] = d_plus
    result_df['topsis_d_minus'] = d_minus
    result_df['topsis_score'] = closeness
    result_df['topsis_rank'] = result_df['topsis_score'].rank(ascending=False).astype(int)

    result_df = result_df.sort_values('topsis_rank')

    logger.info(f"TOPSIS ranking complete. Best: {result_df.iloc[0].name}")

    return result_df


def vikor(
    df: pd.DataFrame,
    criteria: List[str],
    weights: Dict[str, float],
    criteria_types: Dict[str, str],
    v: float = 0.5
) -> pd.DataFrame:
    """
    VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje).

    Args:
        df: DataFrame with alternatives and criteria columns
        criteria: List of criteria column names
        weights: Dict mapping criterion to weight
        criteria_types: Dict mapping criterion to 'benefit' or 'cost'
        v: Weight of maximum group utility (0.5 = compromise)

    Returns:
        DataFrame with VIKOR scores and rankings
    """
    logger = get_logger()

    # Extract decision matrix
    matrix = df[criteria].values.astype(float)
    n_alternatives, n_criteria = matrix.shape

    # Normalize weights
    w = np.array([weights.get(c, 1/n_criteria) for c in criteria])
    w = w / w.sum()

    # Step 1: Determine best (f*) and worst (f-) values for each criterion
    f_star = np.zeros(n_criteria)
    f_minus = np.zeros(n_criteria)

    for j, c in enumerate(criteria):
        col = matrix[:, j]
        if criteria_types.get(c, 'benefit') == 'benefit':
            f_star[j] = np.max(col)
            f_minus[j] = np.min(col)
        else:
            f_star[j] = np.min(col)
            f_minus[j] = np.max(col)

    # Step 2: Compute S (group utility) and R (individual regret)
    S = np.zeros(n_alternatives)
    R = np.zeros(n_alternatives)

    for i in range(n_alternatives):
        for j, c in enumerate(criteria):
            denom = f_star[j] - f_minus[j]
            if denom == 0:
                denom = 1e-10

            if criteria_types.get(c, 'benefit') == 'benefit':
                normalized = (f_star[j] - matrix[i, j]) / denom
            else:
                normalized = (matrix[i, j] - f_star[j]) / denom

            S[i] += w[j] * normalized
            R[i] = max(R[i], w[j] * normalized)

    # Step 3: Compute Q (VIKOR index)
    S_star = np.min(S)
    S_minus = np.max(S)
    R_star = np.min(R)
    R_minus = np.max(R)

    denom_S = S_minus - S_star if S_minus != S_star else 1e-10
    denom_R = R_minus - R_star if R_minus != R_star else 1e-10

    Q = v * (S - S_star) / denom_S + (1 - v) * (R - R_star) / denom_R

    # Create results
    result_df = df.copy()
    result_df['vikor_S'] = S
    result_df['vikor_R'] = R
    result_df['vikor_Q'] = Q
    result_df['vikor_rank'] = result_df['vikor_Q'].rank().astype(int)  # Lower Q is better

    result_df = result_df.sort_values('vikor_rank')

    # Check VIKOR conditions for best alternative
    best_idx = result_df.index[0]
    second_idx = result_df.index[1] if len(result_df) > 1 else best_idx

    Q_best = result_df.loc[best_idx, 'vikor_Q']
    Q_second = result_df.loc[second_idx, 'vikor_Q']

    # Condition 1: Acceptable advantage
    DQ = 1 / (n_alternatives - 1) if n_alternatives > 1 else 0
    acceptable_advantage = (Q_second - Q_best) >= DQ

    # Condition 2: Acceptable stability
    acceptable_stability = (result_df.loc[best_idx, 'vikor_S'] == S_star or
                           result_df.loc[best_idx, 'vikor_R'] == R_star)

    result_df['is_compromise'] = False
    result_df.loc[best_idx, 'is_compromise'] = acceptable_advantage and acceptable_stability

    logger.info(f"VIKOR ranking complete. Best: {result_df.iloc[0].name}")
    logger.info(f"  Acceptable advantage: {acceptable_advantage}")
    logger.info(f"  Acceptable stability: {acceptable_stability}")

    return result_df


def select_best_fs_option(
    eval_matrix: pd.DataFrame,
    config: 'Config',
    use_pareto: bool = True
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Select best FS option using MCDA.

    Args:
        eval_matrix: FS evaluation matrix
        config: Configuration
        use_pareto: Whether to apply Pareto filter first

    Returns:
        Tuple of (best FS option name, ranking DataFrame, selection details)
    """
    logger = get_logger()
    logger.info("Selecting best FS option using MCDA...")

    # Define criteria
    criteria = ['C1_accuracy', 'C2_stability', 'C3_shap_concentration',
                'C4_shap_stability', 'C5_parsimony']

    # Map criteria types (benefit = higher is better, cost = lower is better)
    criteria_types = {
        'C1_accuracy': 'cost',  # Lower MAE is better
        'C2_stability': 'benefit',  # Higher stability is better
        'C3_shap_concentration': 'benefit',  # Higher concentration is better
        'C4_shap_stability': 'benefit',  # Higher SHAP stability is better
        'C5_parsimony': 'benefit'  # Higher parsimony score is better
    }

    # Get weights from config
    weights = {
        'C1_accuracy': config.mcda.fs_weights.get('accuracy', 0.3),
        'C2_stability': config.mcda.fs_weights.get('stability', 0.2),
        'C3_shap_concentration': config.mcda.fs_weights.get('shap_concentration', 0.15),
        'C4_shap_stability': config.mcda.fs_weights.get('shap_stability', 0.2),
        'C5_parsimony': config.mcda.fs_weights.get('parsimony', 0.15)
    }

    # Ensure fs_option is the index
    if 'fs_option' in eval_matrix.columns:
        eval_matrix = eval_matrix.set_index('fs_option')

    # Filter to valid rows
    eval_matrix = eval_matrix.dropna(subset=criteria)

    if len(eval_matrix) == 0:
        raise ValueError("No valid FS options to evaluate")

    # Apply Pareto filter if requested
    working_df = eval_matrix.reset_index()
    if use_pareto and len(working_df) > 1:
        pareto_df = pareto_filter(working_df, criteria, criteria_types)
        working_df = pareto_df

    # Apply MCDA method
    if config.mcda.method == 'vikor':
        ranking_df = vikor(
            working_df, criteria, weights, criteria_types,
            v=config.mcda.vikor_v
        )
        score_col = 'vikor_Q'
        rank_col = 'vikor_rank'
    else:  # topsis
        ranking_df = topsis(working_df, criteria, weights, criteria_types)
        score_col = 'topsis_score'
        rank_col = 'topsis_rank'

    # Get best option
    best_option = ranking_df.iloc[0]['fs_option']

    selection_details = {
        'method': config.mcda.method,
        'use_pareto': use_pareto,
        'weights': weights,
        'criteria_types': criteria_types,
        'n_candidates': len(eval_matrix),
        'n_pareto': len(working_df),
        'best_option': best_option,
        'best_score': float(ranking_df.iloc[0][score_col]),
        'ranking': ranking_df[['fs_option', score_col, rank_col]].to_dict(orient='records')
    }

    logger.info(f"Best FS option: {best_option}")

    return best_option, ranking_df, selection_details


def select_best_model(
    model_metrics: pd.DataFrame,
    config: 'Config',
    use_pareto: bool = True
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Select best forecasting model using MCDA.

    Args:
        model_metrics: Model evaluation metrics DataFrame
        config: Configuration
        use_pareto: Whether to apply Pareto filter first

    Returns:
        Tuple of (best model name, ranking DataFrame, selection details)
    """
    logger = get_logger()
    logger.info("Selecting best model using MCDA...")

    # Define criteria for model selection
    criteria = ['quarterly_mae', 'stability', 'annual_consistency', 'parsimony']

    # Check which criteria exist in the data
    available_criteria = [c for c in criteria if c in model_metrics.columns]

    if len(available_criteria) < 2:
        logger.warning("Not enough criteria available, using available metrics")
        available_criteria = [c for c in model_metrics.columns
                            if c not in ['model', 'model_name'] and
                            model_metrics[c].dtype in [np.float64, np.int64]]

    criteria = available_criteria

    # Map criteria types
    criteria_types = {
        'quarterly_mae': 'cost',
        'weighted_mae': 'cost',
        'mae_h1': 'cost',
        'stability': 'benefit',
        'stability_score': 'benefit',
        'annual_consistency': 'benefit',
        'annual_mae': 'cost',
        'annual_mape': 'cost',
        'parsimony': 'benefit',
        'n_params': 'cost',
        'interpretability': 'benefit'
    }

    # Get weights from config
    default_weight = 1 / len(criteria)
    weights = {}
    for c in criteria:
        if c in config.mcda.model_weights:
            weights[c] = config.mcda.model_weights[c]
        elif c in ['quarterly_mae', 'weighted_mae', 'mae_h1']:
            weights[c] = config.mcda.model_weights.get('quarterly_mae', default_weight)
        elif c in ['stability', 'stability_score']:
            weights[c] = config.mcda.model_weights.get('stability', default_weight)
        elif c in ['annual_consistency', 'annual_mae', 'annual_mape']:
            weights[c] = config.mcda.model_weights.get('annual_consistency', default_weight)
        else:
            weights[c] = default_weight

    # Ensure model name is the index
    if 'model' in model_metrics.columns:
        model_metrics = model_metrics.set_index('model')
    elif 'model_name' in model_metrics.columns:
        model_metrics = model_metrics.set_index('model_name')

    # Filter to valid rows
    model_metrics = model_metrics.dropna(subset=criteria)

    if len(model_metrics) == 0:
        raise ValueError("No valid models to evaluate")

    # Apply Pareto filter
    working_df = model_metrics.reset_index()
    model_col = working_df.columns[0]  # First column is model name

    if use_pareto and len(working_df) > 1:
        pareto_df = pareto_filter(working_df, criteria, criteria_types)
        working_df = pareto_df

    # Apply MCDA method
    if config.mcda.method == 'vikor':
        ranking_df = vikor(
            working_df, criteria, weights, criteria_types,
            v=config.mcda.vikor_v
        )
        score_col = 'vikor_Q'
        rank_col = 'vikor_rank'
    else:
        ranking_df = topsis(working_df, criteria, weights, criteria_types)
        score_col = 'topsis_score'
        rank_col = 'topsis_rank'

    # Get best model
    best_model = ranking_df.iloc[0][model_col]

    selection_details = {
        'method': config.mcda.method,
        'use_pareto': use_pareto,
        'weights': weights,
        'criteria': criteria,
        'criteria_types': {c: criteria_types.get(c, 'benefit') for c in criteria},
        'n_candidates': len(model_metrics),
        'n_pareto': len(working_df),
        'best_model': best_model,
        'best_score': float(ranking_df.iloc[0][score_col]),
        'ranking': ranking_df[[model_col, score_col, rank_col] + criteria].to_dict(orient='records')
    }

    logger.info(f"Best model: {best_model}")

    return best_model, ranking_df, selection_details
