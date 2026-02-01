"""
Sensitivity analysis for MCDA methods (VIKOR/TOPSIS).

Provides functions to analyze ranking stability under parameter changes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from .mcda import vikor, topsis
from ..core.logging_utils import get_logger


def vikor_v_sensitivity(
    df: pd.DataFrame,
    criteria: List[str],
    weights: Dict[str, float],
    criteria_types: Dict[str, str],
    v_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 11
) -> pd.DataFrame:
    """
    Analyze VIKOR ranking sensitivity to v parameter.

    The v parameter controls the balance between group utility (v=1)
    and individual regret (v=0). v=0.5 is the default balanced compromise.

    Args:
        df: DataFrame with alternatives and criteria values
        criteria: List of criteria column names
        weights: Criteria weights (must sum to 1)
        criteria_types: Dict mapping criteria to 'benefit' or 'cost'
        v_range: Range for v parameter (min, max)
        n_points: Number of v values to test

    Returns:
        DataFrame with columns: v, alternative, vikor_Q, rank
    """
    logger = get_logger()
    logger.info(f"Running VIKOR v-parameter sensitivity analysis ({n_points} points)")

    v_values = np.linspace(v_range[0], v_range[1], n_points)
    results = []

    # Get name column
    name_col = 'model' if 'model' in df.columns else 'fs_option'
    if name_col not in df.columns:
        name_col = df.columns[0]

    for v in v_values:
        try:
            ranking = vikor(df.copy(), criteria, weights, criteria_types, v=v)

            for _, row in ranking.iterrows():
                results.append({
                    'v': round(v, 2),
                    'alternative': row[name_col],
                    'vikor_Q': row['vikor_Q'],
                    'vikor_S': row['vikor_S'],
                    'vikor_R': row['vikor_R'],
                    'rank': int(row['vikor_rank'])
                })
        except Exception as e:
            logger.warning(f"VIKOR failed for v={v:.2f}: {e}")
            continue

    result_df = pd.DataFrame(results)
    logger.info(f"Sensitivity analysis complete: {len(result_df)} records")
    return result_df


def weight_sensitivity(
    df: pd.DataFrame,
    criteria: List[str],
    base_weights: Dict[str, float],
    criteria_types: Dict[str, str],
    perturbation: float = 0.2,
    method: str = 'vikor',
    v: float = 0.5
) -> pd.DataFrame:
    """
    Analyze ranking sensitivity to weight perturbations.

    Tests how rankings change when each criterion weight is increased
    or decreased by the perturbation percentage.

    Args:
        df: DataFrame with alternatives and criteria values
        criteria: List of criteria column names
        base_weights: Base criteria weights
        criteria_types: Dict mapping criteria to 'benefit' or 'cost'
        perturbation: Percentage perturbation (0.2 = +/-20%)
        method: MCDA method ('vikor' or 'topsis')
        v: VIKOR v parameter (only used if method='vikor')

    Returns:
        DataFrame with perturbation analysis results
    """
    logger = get_logger()
    logger.info(f"Running weight sensitivity analysis (perturbation={perturbation*100}%)")

    results = []

    # Get name column
    name_col = 'model' if 'model' in df.columns else 'fs_option'
    if name_col not in df.columns:
        name_col = df.columns[0]

    # Base ranking
    try:
        if method == 'vikor':
            base_ranking = vikor(df.copy(), criteria, base_weights, criteria_types, v=v)
            rank_col = 'vikor_rank'
        else:
            base_ranking = topsis(df.copy(), criteria, base_weights, criteria_types)
            rank_col = 'topsis_rank'
    except Exception as e:
        logger.error(f"Base ranking failed: {e}")
        return pd.DataFrame()

    # Store base ranks
    base_ranks = {row[name_col]: int(row[rank_col]) for _, row in base_ranking.iterrows()}

    # Perturb each weight
    for criterion in criteria:
        if criterion not in base_weights:
            continue

        for direction in ['increase', 'decrease']:
            perturbed_weights = base_weights.copy()

            if direction == 'increase':
                perturbed_weights[criterion] *= (1 + perturbation)
            else:
                perturbed_weights[criterion] *= (1 - perturbation)

            # Renormalize weights to sum to 1
            total = sum(perturbed_weights.values())
            perturbed_weights = {k: v/total for k, v in perturbed_weights.items()}

            # Compute ranking
            try:
                if method == 'vikor':
                    perturbed_ranking = vikor(df.copy(), criteria, perturbed_weights,
                                             criteria_types, v=v)
                else:
                    perturbed_ranking = topsis(df.copy(), criteria, perturbed_weights,
                                              criteria_types)

                for _, row in perturbed_ranking.iterrows():
                    alt = row[name_col]
                    new_rank = int(row[rank_col])
                    results.append({
                        'criterion': criterion,
                        'perturbation': direction,
                        'perturbation_pct': f"{'+' if direction == 'increase' else '-'}{int(perturbation*100)}%",
                        'alternative': alt,
                        'base_rank': base_ranks.get(alt, 0),
                        'new_rank': new_rank,
                        'rank_change': new_rank - base_ranks.get(alt, 0)
                    })
            except Exception as e:
                logger.warning(f"Perturbation failed for {criterion}/{direction}: {e}")
                continue

    result_df = pd.DataFrame(results)
    logger.info(f"Weight sensitivity complete: {len(result_df)} records")
    return result_df


def criterion_removal_sensitivity(
    df: pd.DataFrame,
    criteria: List[str],
    base_weights: Dict[str, float],
    criteria_types: Dict[str, str],
    method: str = 'vikor',
    v: float = 0.5
) -> pd.DataFrame:
    """
    Analyze ranking sensitivity to removing individual criteria.

    Tests rank stability by removing one criterion at a time.

    Args:
        df: DataFrame with alternatives and criteria values
        criteria: List of criteria column names
        base_weights: Base criteria weights
        criteria_types: Dict mapping criteria to 'benefit' or 'cost'
        method: MCDA method ('vikor' or 'topsis')
        v: VIKOR v parameter

    Returns:
        DataFrame with criterion removal analysis results
    """
    logger = get_logger()
    logger.info("Running criterion removal sensitivity analysis")

    results = []

    # Get name column
    name_col = 'model' if 'model' in df.columns else 'fs_option'
    if name_col not in df.columns:
        name_col = df.columns[0]

    # Base ranking
    try:
        if method == 'vikor':
            base_ranking = vikor(df.copy(), criteria, base_weights, criteria_types, v=v)
            rank_col = 'vikor_rank'
        else:
            base_ranking = topsis(df.copy(), criteria, base_weights, criteria_types)
            rank_col = 'topsis_rank'
    except Exception as e:
        logger.error(f"Base ranking failed: {e}")
        return pd.DataFrame()

    base_ranks = {row[name_col]: int(row[rank_col]) for _, row in base_ranking.iterrows()}

    # Remove each criterion one at a time
    for removed_criterion in criteria:
        reduced_criteria = [c for c in criteria if c != removed_criterion]
        reduced_weights = {k: v for k, v in base_weights.items() if k != removed_criterion}
        reduced_types = {k: v for k, v in criteria_types.items() if k != removed_criterion}

        # Renormalize weights
        total = sum(reduced_weights.values())
        if total > 0:
            reduced_weights = {k: v/total for k, v in reduced_weights.items()}

        try:
            if method == 'vikor':
                reduced_ranking = vikor(df.copy(), reduced_criteria, reduced_weights,
                                       reduced_types, v=v)
            else:
                reduced_ranking = topsis(df.copy(), reduced_criteria, reduced_weights,
                                        reduced_types)

            for _, row in reduced_ranking.iterrows():
                alt = row[name_col]
                new_rank = int(row[rank_col])
                results.append({
                    'removed_criterion': removed_criterion,
                    'alternative': alt,
                    'base_rank': base_ranks.get(alt, 0),
                    'new_rank': new_rank,
                    'rank_change': new_rank - base_ranks.get(alt, 0),
                    'rank_reversed': (base_ranks.get(alt, 0) == 1) != (new_rank == 1)
                })
        except Exception as e:
            logger.warning(f"Removal analysis failed for {removed_criterion}: {e}")
            continue

    result_df = pd.DataFrame(results)
    logger.info(f"Criterion removal sensitivity complete: {len(result_df)} records")
    return result_df


def compute_rank_stability_score(sensitivity_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute overall rank stability scores from sensitivity analysis.

    Args:
        sensitivity_df: DataFrame from weight_sensitivity() or vikor_v_sensitivity()

    Returns:
        Dict mapping alternative to stability score (0-1, higher is more stable)
    """
    if 'rank_change' not in sensitivity_df.columns:
        return {}

    stability_scores = {}
    for alt in sensitivity_df['alternative'].unique():
        alt_data = sensitivity_df[sensitivity_df['alternative'] == alt]
        # Stability = proportion of scenarios with no rank change
        no_change = (alt_data['rank_change'] == 0).sum()
        total = len(alt_data)
        stability_scores[alt] = no_change / total if total > 0 else 0

    return stability_scores
