"""
High-resolution plotting for CO2 forecasting framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Set high-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.figsize'] = (12, 8)
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

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontweight='bold', fontsize=14)
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
