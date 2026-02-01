"""
High-resolution plotting for CO2 forecasting framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Set high-quality defaults with BIGGER, BOLDER text
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def set_plot_style(style: str = 'seaborn-v0_8-whitegrid'):
    """Set matplotlib style."""
    try:
        plt.style.use(style)
    except:
        plt.style.use('seaborn-whitegrid')


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    title: str = "Predictions vs Actual",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot predictions vs actual values over time.

    Args:
        predictions_df: DataFrame with 'date', 'actual', 'predicted' columns
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Main plot
    ax1 = axes[0]
    dates = pd.to_datetime(predictions_df['date'])

    ax1.plot(dates, predictions_df['actual'], 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(dates, predictions_df['predicted'], 'r--', label='Predicted', linewidth=2, alpha=0.8)

    ax1.fill_between(dates, predictions_df['actual'], predictions_df['predicted'],
                     alpha=0.2, color='gray')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_title(title, fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # Error plot
    ax2 = axes[1]
    errors = predictions_df['actual'] - predictions_df['predicted']
    ax2.bar(dates, errors, color='steelblue', alpha=0.7, width=60)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Errors', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_horizon_comparison(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    metric_name: str = 'mae',
    title: str = "Performance by Forecast Horizon",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot metrics comparison across forecast horizons.

    Args:
        metrics_by_horizon: Dict mapping horizon to metrics dict
        metric_name: Metric to plot
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    horizons = sorted(metrics_by_horizon.keys())
    values = [metrics_by_horizon[h].get(metric_name, 0) for h in horizons]

    bars = ax.bar([f'{h}Q' for h in horizons], values, color='steelblue', alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['weighted_mae', 'stability_score'],
    title: str = "Model Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot comparison of multiple models across metrics.

    Args:
        comparison_df: DataFrame with model comparisons
        metrics: List of metrics to compare
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric not in comparison_df.columns:
            continue

        values = comparison_df[metric].values
        models = comparison_df['model'].values if 'model' in comparison_df.columns else comparison_df.index

        bars = ax.barh(models, values, color=colors, edgecolor='black', alpha=0.8)

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01*max(values), bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', ha='left', va='center', fontsize=9)

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_optimization_history(
    history: Dict[str, List],
    title: str = "Optimization History",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot optimization convergence history.

    Args:
        history: Optimization history dict
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = history['iterations']
    best_fitness = history['best_fitness']

    ax.plot(iterations, best_fitness, 'b-', linewidth=2, marker='o',
            markersize=4, label='Best Fitness')

    if 'mean_fitness' in history:
        ax.plot(iterations, history['mean_fitness'], 'g--', linewidth=1.5,
                alpha=0.7, label='Mean Fitness')
        ax.fill_between(iterations, best_fitness, history['mean_fitness'],
                       alpha=0.2, color='green')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness (Weighted MAE)')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    # Annotate best value
    best_idx = np.argmin(best_fitness)
    ax.annotate(f'Best: {best_fitness[best_idx]:.4f}',
               xy=(iterations[best_idx], best_fitness[best_idx]),
               xytext=(iterations[best_idx] + 2, best_fitness[best_idx] * 1.05),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_pareto_front(
    pareto_df: pd.DataFrame,
    obj1: str = 'weighted_mae',
    obj2: str = 'stability_score',
    all_points_df: Optional[pd.DataFrame] = None,
    title: str = "Pareto Front",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot Pareto front for multi-objective optimization.

    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        obj1: First objective (x-axis)
        obj2: Second objective (y-axis)
        all_points_df: Optional DataFrame with all solutions (dominated)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot dominated points
    if all_points_df is not None:
        dominated = all_points_df[~all_points_df.index.isin(pareto_df.index)]
        ax.scatter(dominated[obj1], dominated[obj2], c='gray', alpha=0.5,
                  s=100, label='Dominated', marker='o')

    # Plot Pareto front
    pareto_sorted = pareto_df.sort_values(obj1)
    ax.scatter(pareto_sorted[obj1], pareto_sorted[obj2], c='red',
              s=150, label='Pareto Front', marker='*', edgecolors='black', linewidths=1)
    ax.plot(pareto_sorted[obj1], pareto_sorted[obj2], 'r--', alpha=0.5, linewidth=2)

    # Annotate points
    for idx, row in pareto_df.iterrows():
        label = row.get('model', row.get('fs_option', str(idx)))
        ax.annotate(label, (row[obj1], row[obj2]), textcoords="offset points",
                   xytext=(5, 5), fontsize=9)

    ax.set_xlabel(obj1.replace('_', ' ').title())
    ax.set_ylabel(obj2.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_mcda_ranking(
    ranking_df: pd.DataFrame,
    score_col: str = 'vikor_Q',
    title: str = "MCDA Ranking (VIKOR)",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot MCDA ranking results.

    Args:
        ranking_df: DataFrame with MCDA results
        score_col: Column with MCDA scores
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by score
    df = ranking_df.sort_values(score_col)

    # Get names
    name_col = 'model' if 'model' in df.columns else 'fs_option'
    if name_col not in df.columns:
        name_col = df.columns[0]

    names = df[name_col].values
    scores = df[score_col].values

    # Create color gradient (best = green, worst = red)
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(names)))

    bars = ax.barh(names, scores, color=colors, edgecolor='black', alpha=0.8)

    # Add rank labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'#{i+1} ({score:.4f})', ha='left', va='center', fontsize=10)

    ax.set_xlabel('MCDA Score (lower is better)' if 'vikor' in score_col.lower() else 'MCDA Score')
    ax.set_title(title, fontweight='bold')

    # Highlight best
    ax.axvline(x=scores[0], color='green', linestyle='--', alpha=0.5, linewidth=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort and select top N
    df = importance_df.nlargest(top_n, 'importance')

    # Create gradient colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]

    bars = ax.barh(df['feature'].values[::-1], df['importance'].values[::-1],
                   color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f'{bar.get_width():.4f}', ha='left', va='center', fontsize=9)

    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    title: str = "SHAP Summary",
    output_path: Optional[Path] = None,
    max_display: int = 15
):
    """
    Create SHAP summary plot.

    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        title: Plot title
        output_path: Path to save figure
        max_display: Maximum features to display
    """
    import shap

    fig = plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontweight='bold', fontsize=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    title: str = "SHAP Beeswarm",
    output_path: Optional[Path] = None,
    max_display: int = 15,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create SHAP beeswarm plot (modern replacement for summary_plot).

    Args:
        shap_values: SHAP values (numpy array or shap.Explanation object)
        X: Feature DataFrame
        title: Plot title
        output_path: Path to save figure
        max_display: Maximum features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import shap

    plt.figure(figsize=figsize)

    # Create Explanation object if necessary
    if not isinstance(shap_values, shap.Explanation):
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.zeros(len(shap_values)),
            data=X.values,
            feature_names=list(X.columns)
        )
    else:
        explanation = shap_values

    # Use beeswarm plot
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)

    plt.title(title, fontweight='bold', fontsize=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return plt.gcf()


def plot_model_coefficients(
    coef_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 15,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot model coefficients or feature importance with color coding.

    For linear models: positive coefficients in green, negative in red.
    For tree models: gradient blue colors for importance.

    Args:
        coef_df: DataFrame with 'feature' and coefficient/importance columns
        model_name: Name of the model for title
        top_n: Number of top features to show
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine which column to use
    if 'coefficient' in coef_df.columns:
        value_col = 'coefficient'
        sort_col = 'abs_coefficient' if 'abs_coefficient' in coef_df.columns else 'coefficient'
        df = coef_df.nlargest(top_n, sort_col).copy()
        values = df[value_col].values
        # Color by sign: positive=green, negative=red
        colors = ['forestgreen' if v > 0 else 'crimson' for v in values]
        xlabel = 'Coefficient Value'
    elif 'importance' in coef_df.columns:
        value_col = 'importance'
        df = coef_df.nlargest(top_n, 'importance').copy()
        values = df['importance'].values
        # Gradient blue colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]
        xlabel = 'Feature Importance'
    else:
        # Fallback
        value_col = coef_df.columns[1]
        df = coef_df.nlargest(top_n, value_col).copy()
        values = df[value_col].values
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]
        xlabel = value_col.replace('_', ' ').title()

    # Create horizontal bar plot
    bars = ax.barh(df['feature'].values[::-1], np.abs(values)[::-1] if value_col == 'coefficient' else values[::-1],
                   color=colors[::-1] if isinstance(colors, list) else colors,
                   edgecolor='black', alpha=0.85, linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, (values[::-1] if value_col == 'coefficient' else values[::-1])):
        label = f'{val:+.4f}' if value_col == 'coefficient' else f'{val:.4f}'
        ax.text(bar.get_width() + 0.005 * max(np.abs(values)), bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=16)
    ax.set_ylabel('Feature', fontweight='bold', fontsize=16)
    ax.set_title(f'{model_name} - Feature Coefficients/Importance', fontweight='bold', fontsize=20)

    # Add legend for coefficient plots
    if value_col == 'coefficient':
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='forestgreen', label='Positive'),
                          Patch(facecolor='crimson', label='Negative')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_vikor_radar(
    ranking_df: pd.DataFrame,
    criteria: List[str],
    top_n: int = 5,
    title: str = "VIKOR Criteria Comparison (Radar Chart)",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create radar/spider chart for VIKOR alternative comparison.

    Args:
        ranking_df: DataFrame with VIKOR results (must have criteria columns)
        criteria: List of criteria column names
        top_n: Number of top alternatives to display
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from math import pi

    # Get top N alternatives
    df = ranking_df.head(top_n).copy()

    # Get name column
    name_col = 'model' if 'model' in df.columns else 'fs_option'
    if name_col not in df.columns:
        name_col = df.columns[0]

    # Filter to available criteria
    available_criteria = [c for c in criteria if c in df.columns]
    if not available_criteria:
        print(f"Warning: No criteria found in DataFrame. Available columns: {df.columns.tolist()}")
        return None

    # Normalize criteria values to 0-1 range for radar plot
    norm_df = df[available_criteria].copy()
    for col in available_criteria:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5

    # Number of criteria
    N = len(available_criteria)

    # Angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Color palette (rank-based: best = green, worst = red)
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, top_n))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = norm_df.iloc[idx][available_criteria].values.tolist()
        values += values[:1]  # Close the polygon

        label = f"#{idx+1}: {row[name_col]}"
        ax.plot(angles, values, 'o-', linewidth=2.5, label=label, color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Set axis labels with better formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', '\n').replace('weighted', 'wtd').title()
                        for c in available_criteria],
                       fontsize=12, fontweight='bold')

    ax.set_title(title, fontweight='bold', fontsize=20, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_vikor_v_sensitivity(
    sensitivity_df: pd.DataFrame,
    title: str = "VIKOR Ranking Sensitivity to v Parameter",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot ranking changes across v parameter values.

    Args:
        sensitivity_df: DataFrame with 'v', 'alternative', 'rank' columns
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    alternatives = sensitivity_df['alternative'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(alternatives)))

    for idx, alt in enumerate(alternatives):
        data = sensitivity_df[sensitivity_df['alternative'] == alt]
        ax.plot(data['v'], data['rank'], 'o-', label=alt,
                linewidth=2.5, markersize=10, color=colors[idx])

    ax.set_xlabel('VIKOR v Parameter (0=Regret Focus, 1=Utility Focus)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Rank (1=Best)', fontweight='bold', fontsize=16)
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
    ax.invert_yaxis()  # Lower rank is better
    ax.set_yticks(range(1, len(alternatives) + 1))
    ax.grid(True, alpha=0.3)

    # Add vertical line at v=0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.annotate('Balanced\n(v=0.5)', xy=(0.5, 1), xytext=(0.55, 1.5),
                fontsize=11, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_weight_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    title: str = "Rank Stability Under Weight Perturbations",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot heatmap of rank changes under weight perturbations.

    Args:
        sensitivity_df: DataFrame with perturbation results
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn required for heatmap. Install with: pip install seaborn")
        return None

    # Pivot to get rank changes
    pivot = sensitivity_df.pivot_table(
        index='alternative',
        columns=['criterion', 'perturbation'],
        values='rank_change',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                center=0, ax=ax, cbar_kws={'label': 'Rank Change'},
                annot_kws={'size': 11, 'weight': 'bold'},
                linewidths=0.5, linecolor='white')

    ax.set_title(title, fontweight='bold', fontsize=20, pad=15)
    ax.set_xlabel('Criterion & Perturbation Direction', fontweight='bold', fontsize=14)
    ax.set_ylabel('Alternative', fontweight='bold', fontsize=14)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_pareto_front_enhanced(
    pareto_df: pd.DataFrame,
    obj1: str = 'weighted_mae',
    obj2: str = 'n_features',
    all_points_df: Optional[pd.DataFrame] = None,
    title: str = "Pareto Front: Predictive Performance vs. Computational Footprint",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
    x_label: str = None,
    y_label: str = None,
    annotate_info: bool = True
) -> plt.Figure:
    """
    Enhanced Pareto front plot with meaningful annotations.

    Args:
        pareto_df: DataFrame with Pareto-optimal solutions
        obj1: First objective (x-axis)
        obj2: Second objective (y-axis)
        all_points_df: Optional DataFrame with all solutions
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        x_label: Custom x-axis label
        y_label: Custom y-axis label
        annotate_info: Whether to add detailed annotations

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine labels
    if x_label is None:
        if 'mae' in obj1.lower() or 'accuracy' in obj1.lower():
            x_label = "Predictive Performance (lower error = better)"
        else:
            x_label = obj1.replace('_', ' ').title()

    if y_label is None:
        if 'n_features' in obj2.lower() or 'parsimony' in obj2.lower():
            y_label = "Model Simplicity (fewer features = lower footprint)"
        elif 'stability' in obj2.lower():
            y_label = "Prediction Stability (higher = more robust)"
        else:
            y_label = obj2.replace('_', ' ').title()

    # Plot dominated points
    if all_points_df is not None and obj1 in all_points_df.columns and obj2 in all_points_df.columns:
        dominated = all_points_df[~all_points_df.index.isin(pareto_df.index)]
        if len(dominated) > 0:
            ax.scatter(dominated[obj1], dominated[obj2], c='lightgray', alpha=0.6,
                      s=150, label='Dominated Solutions', marker='o', edgecolors='gray', linewidths=1)

    # Plot Pareto front with color gradient
    if obj1 in pareto_df.columns and obj2 in pareto_df.columns:
        pareto_sorted = pareto_df.sort_values(obj1)
        n_points = len(pareto_sorted)
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, n_points))

        ax.scatter(pareto_sorted[obj1], pareto_sorted[obj2], c=colors,
                  s=250, label='Pareto Optimal', marker='*', edgecolors='black', linewidths=1.5)
        ax.plot(pareto_sorted[obj1], pareto_sorted[obj2], 'k--', alpha=0.4, linewidth=2)

        # Annotate points with meaningful info
        name_col = 'model' if 'model' in pareto_df.columns else 'fs_option'
        if name_col not in pareto_df.columns:
            name_col = pareto_df.columns[0]

        for idx, (_, row) in enumerate(pareto_sorted.iterrows()):
            label = str(row.get(name_col, idx))

            # Build annotation text
            info_parts = [f"#{idx+1}: {label}"]
            if annotate_info:
                if 'n_features' in row.index:
                    info_parts.append(f"Features: {int(row['n_features'])}")
                if 'weighted_mae' in row.index:
                    info_parts.append(f"MAE: {row['weighted_mae']:.4f}")

            annotation = '\n'.join(info_parts)

            ax.annotate(annotation, (row[obj1], row[obj2]),
                       textcoords="offset points",
                       xytext=(10, 10), fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel(x_label, fontweight='bold', fontsize=16)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=16)
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add interpretive text box
    textstr = "Trade-off Region:\n" \
              "Lower-left = Best (low error, low complexity)\n" \
              "Upper-right = Worst (high error, high complexity)"
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', fontweight='bold', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_annual_consistency(
    annual_df: pd.DataFrame,
    title: str = "Annual Consistency: Predicted vs Actual",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot annual predicted vs actual values.

    Args:
        annual_df: DataFrame with 'year', 'actual_annual', 'predicted_annual'
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # Main comparison plot
    ax1 = axes[0]
    years = annual_df['year'].values

    width = 0.35
    x = np.arange(len(years))

    bars1 = ax1.bar(x - width/2, annual_df['actual_annual'], width,
                   label='Actual', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, annual_df['predicted_annual'], width,
                   label='Predicted', color='coral', alpha=0.8)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual CO2 Emissions')
    ax1.set_title(title, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45)
    ax1.legend()

    # Error plot
    ax2 = axes[1]
    errors = annual_df['actual_annual'] - annual_df['predicted_annual']
    error_pct = (errors / annual_df['actual_annual']) * 100

    colors = ['green' if e >= 0 else 'red' for e in errors]
    ax2.bar(x, error_pct, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Annual Prediction Error (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_regime_comparison(
    regime_data: Dict[str, pd.DataFrame],
    title: str = "Feature Importance by Regime",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    top_n: int = 10
) -> plt.Figure:
    """
    Plot feature importance comparison across regimes.

    Args:
        regime_data: Dict mapping regime name to importance DataFrame
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        top_n: Number of top features to show

    Returns:
        Matplotlib figure
    """
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=figsize, sharey=True)

    if n_regimes == 1:
        axes = [axes]

    colors = {'pre_covid': 'steelblue', 'covid': 'coral', 'post_covid': 'green',
              'energy_crisis': 'purple'}

    for idx, (regime_name, df) in enumerate(regime_data.items()):
        ax = axes[idx]
        df_top = df.nlargest(top_n, 'importance')

        color = colors.get(regime_name, 'gray')
        ax.barh(df_top['feature'].values[::-1], df_top['importance'].values[::-1],
               color=color, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Importance')
        ax.set_title(regime_name.replace('_', ' ').title(), fontweight='bold')

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def create_all_plots(
    results: Dict[str, Any],
    output_dir: Path,
    config: Any = None
):
    """
    Create all standard plots for a model evaluation run.

    Args:
        results: Dictionary with all results
        output_dir: Directory to save plots
        config: Configuration object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create available plots based on results
    if 'predictions' in results:
        plot_predictions_vs_actual(
            results['predictions'],
            output_path=output_dir / 'predictions_vs_actual.png'
        )

    if 'horizon_metrics' in results:
        plot_horizon_comparison(
            results['horizon_metrics'],
            output_path=output_dir / 'horizon_comparison.png'
        )

    if 'model_comparison' in results:
        plot_model_comparison(
            results['model_comparison'],
            output_path=output_dir / 'model_comparison.png'
        )

    if 'optimization_history' in results:
        for model, history in results['optimization_history'].items():
            plot_optimization_history(
                history,
                title=f"Optimization History - {model}",
                output_path=output_dir / f'optimization_{model}.png'
            )

    if 'pareto_front' in results:
        plot_pareto_front(
            results['pareto_front'],
            output_path=output_dir / 'pareto_front.png'
        )

    if 'mcda_ranking' in results:
        plot_mcda_ranking(
            results['mcda_ranking'],
            output_path=output_dir / 'mcda_ranking.png'
        )

    if 'feature_importance' in results:
        plot_feature_importance(
            pd.DataFrame(results['feature_importance']),
            output_path=output_dir / 'feature_importance.png'
        )

    if 'annual_consistency' in results:
        plot_annual_consistency(
            pd.DataFrame(results['annual_consistency']),
            output_path=output_dir / 'annual_consistency.png'
        )

    plt.close('all')


def plot_fs_model_comparison_heatmap(
    metrics_df: pd.DataFrame,
    metric: str = 'weighted_mae',
    title: str = "Model Performance Across FS Options",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot heatmap comparing model performance across FS options.

    Args:
        metrics_df: DataFrame with 'model', 'fs_option', and metric columns
        metric: Metric to visualize
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn required. Install with: pip install seaborn")
        return None

    # Pivot to create heatmap data
    heatmap_data = metrics_df.pivot_table(
        index='model', columns='fs_option', values=metric, aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap direction
    is_lower_better = 'mae' in metric.lower() or 'error' in metric.lower() or 'mape' in metric.lower()
    cmap = 'RdYlGn_r' if is_lower_better else 'RdYlGn'

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap=cmap, ax=ax,
                annot_kws={'size': 11, 'weight': 'bold'},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': metric.replace('_', ' ').title()})

    ax.set_title(title, fontweight='bold', fontsize=18, pad=15)
    ax.set_xlabel('Feature Selection Option', fontweight='bold', fontsize=14)
    ax.set_ylabel('Model', fontweight='bold', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)

    # Highlight best values
    best_per_col = heatmap_data.idxmin() if is_lower_better else heatmap_data.idxmax()
    for col_idx, (col, best_model) in enumerate(best_per_col.items()):
        row_idx = list(heatmap_data.index).index(best_model)
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                                   fill=False, edgecolor='gold', lw=3))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_fs_comparison_bars(
    metrics_df: pd.DataFrame,
    metric: str = 'weighted_mae',
    title: str = "Model Comparison Across FS Options",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot grouped bar chart comparing models across FS options.

    Args:
        metrics_df: DataFrame with 'model', 'fs_option', and metric columns
        metric: Metric to visualize
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = metrics_df['model'].unique()
    fs_options = metrics_df['fs_option'].unique()

    x = np.arange(len(fs_options))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df['model'] == model]
        values = []
        for fs in fs_options:
            fs_data = model_data[model_data['fs_option'] == fs]
            if len(fs_data) > 0:
                values.append(fs_data[metric].values[0])
            else:
                values.append(np.nan)

        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i],
                     edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Feature Selection Option', fontweight='bold', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=18)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(fs_options, rotation=45, ha='right', fontsize=11)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_best_per_fs_option(
    metrics_df: pd.DataFrame,
    metric: str = 'weighted_mae',
    title: str = "Best Model Performance per FS Option",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot best model for each FS option.

    Args:
        metrics_df: DataFrame with 'model', 'fs_option', and metric columns
        metric: Metric to visualize (lower is better assumed)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get best model per FS option
    is_lower_better = 'mae' in metric.lower() or 'error' in metric.lower() or 'mape' in metric.lower()
    if is_lower_better:
        best_idx = metrics_df.groupby('fs_option')[metric].idxmin()
    else:
        best_idx = metrics_df.groupby('fs_option')[metric].idxmax()

    best_per_fs = metrics_df.loc[best_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by model
    unique_models = best_per_fs['model'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_models)))
    model_colors = dict(zip(unique_models, colors))

    bar_colors = [model_colors[m] for m in best_per_fs['model']]
    bars = ax.bar(best_per_fs['fs_option'], best_per_fs[metric],
                 color=bar_colors, edgecolor='black', linewidth=1)

    # Annotate with model names
    for bar, model_name, val in zip(bars, best_per_fs['model'], best_per_fs[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{model_name}\n({val:.4f})', ha='center', va='bottom',
               fontsize=10, fontweight='bold')

    ax.set_xlabel('Feature Selection Option', fontweight='bold', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add legend for model colors
    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
                     for c in model_colors.values()]
    ax.legend(legend_patches, model_colors.keys(), title='Best Model',
             loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_multi_fs_summary(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive multi-panel summary of FS-model comparison.

    Args:
        metrics_df: DataFrame with 'model', 'fs_option', and metric columns
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn required. Install with: pip install seaborn")
        return None

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    heatmap_data = metrics_df.pivot_table(
        index='model', columns='fs_option', values='weighted_mae', aggfunc='first'
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax1,
                annot_kws={'size': 9}, linewidths=0.5)
    ax1.set_title('Weighted MAE by Model & FS Option', fontweight='bold', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax1.get_yticklabels(), fontsize=10)

    # Panel 2: Best model per FS
    ax2 = fig.add_subplot(gs[0, 1])
    best_idx = metrics_df.groupby('fs_option')['weighted_mae'].idxmin()
    best_per_fs = metrics_df.loc[best_idx]
    colors = plt.cm.Set2(np.linspace(0, 1, len(best_per_fs['model'].unique())))
    model_color_map = dict(zip(best_per_fs['model'].unique(), colors))
    bar_colors = [model_color_map[m] for m in best_per_fs['model']]
    ax2.bar(best_per_fs['fs_option'], best_per_fs['weighted_mae'],
           color=bar_colors, edgecolor='black')
    for i, (fs, model, val) in enumerate(zip(best_per_fs['fs_option'],
                                             best_per_fs['model'],
                                             best_per_fs['weighted_mae'])):
        ax2.text(i, val, f'{model}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title('Best Model per FS Option', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Weighted MAE', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Model consistency across FS options (box plot)
    ax3 = fig.add_subplot(gs[1, 0])
    model_order = metrics_df.groupby('model')['weighted_mae'].mean().sort_values().index
    sns.boxplot(data=metrics_df, x='model', y='weighted_mae', order=model_order, ax=ax3,
                palette='Set2')
    ax3.set_title('Model Consistency Across FS Options', fontweight='bold', fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('Weighted MAE', fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: FS option summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    fs_summary = metrics_df.groupby('fs_option').agg({
        'weighted_mae': ['mean', 'std', 'min'],
        'n_features': 'first'
    })
    fs_summary.columns = ['Mean MAE', 'Std MAE', 'Best MAE', 'Features']
    fs_summary = fs_summary.sort_values('Best MAE')

    ax4.barh(fs_summary.index, fs_summary['Best MAE'], color='steelblue',
            alpha=0.7, label='Best MAE')
    ax4.barh(fs_summary.index, fs_summary['Mean MAE'], color='coral',
            alpha=0.5, label='Mean MAE')
    ax4.errorbar(fs_summary['Mean MAE'], fs_summary.index, xerr=fs_summary['Std MAE'],
                fmt='none', color='black', capsize=3)

    # Add feature count as text
    for i, (fs, row) in enumerate(fs_summary.iterrows()):
        ax4.text(row['Best MAE'] + 0.001, i, f"({int(row['Features'])} feat)",
                va='center', fontsize=9)

    ax4.set_title('FS Option Summary', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Weighted MAE', fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(axis='x', alpha=0.3)

    fig.suptitle('Multi-FS Model Comparison Summary', fontweight='bold', fontsize=18, y=1.02)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig
