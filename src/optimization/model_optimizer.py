"""
Model hyperparameter optimization using swarm optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from ..core.logging_utils import get_logger
from ..core.config import Config
from ..core.utils import calculate_weighted_mae, save_json_numpy
from ..splits.walk_forward import CVPlan, generate_cv_folds
from ..models.base import ModelRegistry
from ..models.traditional import get_model_param_space
from ..models.lstm import get_lstm_param_space
from .pso import PSO, GWO, get_optimizer


def create_objective_function(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    model_name: str,
    param_names: List[str],
    param_types: List[str],
    bounds: List[Tuple[float, float]],
    config: Config
) -> callable:
    """
    Create objective function for optimization.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        model_name: Name of the model
        param_names: List of parameter names
        param_types: List of parameter types
        bounds: Parameter bounds
        config: Configuration

    Returns:
        Objective function that takes position array and returns fitness
    """
    logger = get_logger()
    horizon_weights = config.splits.horizon_weights

    def objective(position: np.ndarray) -> float:
        # Convert position to parameters
        params = {}
        for i, (name, ptype, value) in enumerate(zip(param_names, param_types, position)):
            if ptype == 'integer':
                params[name] = int(round(value))
            elif ptype == 'log':
                params[name] = float(value)
            elif ptype == 'categorical':
                # Handle categorical by index
                params[name] = int(round(value))
            else:
                params[name] = float(value)

        # Evaluate with walk-forward CV
        horizon_errors = {h: [] for h in config.splits.horizons}

        try:
            for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
                # Create and train model
                model = ModelRegistry.create(model_name, params)
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate MAE
                mae = np.mean(np.abs(y_test.values - y_pred))
                horizon_errors[fold.horizon].append(mae)

        except Exception as e:
            logger.warning(f"Model training failed with params {params}: {e}")
            return float('inf')

        # Calculate weighted MAE
        mean_errors = {}
        for h in config.splits.horizons:
            if horizon_errors[h]:
                mean_errors[h] = np.mean(horizon_errors[h])
            else:
                mean_errors[h] = float('inf')

        weighted_mae = calculate_weighted_mae(mean_errors, horizon_weights)

        # Optional: add annual consistency penalty
        # (simplified - full implementation in evaluation module)
        if hasattr(config.optimization, 'annual_penalty_weight'):
            penalty_weight = config.optimization.annual_penalty_weight
            if penalty_weight > 0:
                # Add small penalty for unstable models
                stability_penalty = np.std(list(mean_errors.values()))
                weighted_mae += penalty_weight * stability_penalty

        return weighted_mae

    return objective


def optimize_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    model_name: str,
    config: Config,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a single model.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        model_name: Name of the model
        config: Configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results
    """
    logger = get_logger()
    logger.info(f"Optimizing {model_name}...")

    # Get parameter space
    if model_name == 'lstm':
        param_space = get_lstm_param_space()
    else:
        param_space = get_model_param_space(model_name)

    if not param_space:
        logger.warning(f"No parameter space defined for {model_name}, using defaults")
        return {
            'model': model_name,
            'best_params': {},
            'best_fitness': None,
            'history': None
        }

    # Build bounds, names, and types
    param_names = []
    param_types = []
    bounds = []

    for name, spec in param_space.items():
        param_names.append(name)

        if spec['type'] == 'int':
            param_types.append('integer')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'log_uniform':
            param_types.append('continuous')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'uniform':
            param_types.append('continuous')
            bounds.append((spec['low'], spec['high']))
        elif spec['type'] == 'categorical':
            param_types.append('integer')
            bounds.append((0, len(spec['choices']) - 1))
        else:
            param_types.append('continuous')
            bounds.append((spec.get('low', 0), spec.get('high', 1)))

    # Create objective function
    objective = create_objective_function(
        X, y, cv_plan, model_name,
        param_names, param_types, bounds, config
    )

    # Get optimizer
    optimizer = get_optimizer(
        config.optimization.optimizer,
        n_particles=config.optimization.n_particles,
        n_iterations=config.optimization.n_iterations,
        seed=config.seed
    )

    # Run optimization
    best_position, best_fitness, history = optimizer.optimize(
        objective, bounds, param_types
    )

    # Convert position to parameters
    best_params = {}
    for i, (name, ptype, value) in enumerate(zip(param_names, param_types, best_position)):
        spec = param_space[name]
        if ptype == 'integer':
            if spec['type'] == 'categorical':
                best_params[name] = spec['choices'][int(round(value))]
            else:
                best_params[name] = int(round(value))
        else:
            best_params[name] = float(value)

    results = {
        'model': model_name,
        'best_params': best_params,
        'best_fitness': float(best_fitness),
        'history': {
            'iterations': history['iterations'],
            'best_fitness': [float(f) for f in history['best_fitness']],
            'mean_fitness': [float(f) for f in history['mean_fitness']]
        },
        'param_space': param_space,
        'optimizer': config.optimization.optimizer
    }

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json_numpy(results, output_dir / f"{model_name}_optimization.json")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best fitness: {best_fitness:.6f}")

    return results


def optimize_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_plan: CVPlan,
    config: Config,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize hyperparameters for all models.

    Args:
        X: Feature DataFrame
        y: Target Series
        cv_plan: CV plan
        config: Configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results for all models
    """
    logger = get_logger()
    logger.info("Optimizing all models...")

    results = {}

    for model_name in config.model.models:
        try:
            model_results = optimize_model(
                X, y, cv_plan, model_name, config, output_dir
            )
            results[model_name] = model_results
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            results[model_name] = {
                'model': model_name,
                'error': str(e)
            }

    # Save summary
    if output_dir:
        summary = []
        for name, res in results.items():
            summary.append({
                'model': name,
                'best_fitness': res.get('best_fitness'),
                'best_params': res.get('best_params', {})
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / 'optimization_summary.csv', index=False)

    return results


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    best_params: Dict[str, Any]
) -> 'BaseForecaster':
    """
    Train a model with optimized parameters.

    Args:
        X_train: Training features
        y_train: Training target
        model_name: Model name
        best_params: Optimized parameters

    Returns:
        Trained model
    """
    model = ModelRegistry.create(model_name, best_params)
    model.fit(X_train, y_train)
    return model
