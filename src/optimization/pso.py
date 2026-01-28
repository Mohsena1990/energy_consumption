"""
Particle Swarm Optimization (PSO) for hyperparameter tuning.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
import copy

from ..core.logging_utils import get_logger


@dataclass
class Particle:
    """Represents a particle in PSO."""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray = None
    best_fitness: float = float('inf')
    current_fitness: float = float('inf')

    def __post_init__(self):
        if self.best_position is None:
            self.best_position = self.position.copy()


@dataclass
class SwarmOptimizer:
    """Base class for swarm optimization."""
    n_particles: int = 20
    n_iterations: int = 30
    seed: int = 42

    def optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        param_types: List[str] = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Optimize the objective function.

        Args:
            objective: Function to minimize
            bounds: List of (min, max) tuples for each parameter
            param_types: List of parameter types ('continuous', 'integer', 'log')

        Returns:
            Tuple of (best_position, best_fitness, history)
        """
        raise NotImplementedError


class PSO(SwarmOptimizer):
    """
    Particle Swarm Optimization implementation.

    Standard PSO with inertia weight and cognitive/social parameters.
    """

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float = 0.99,
        seed: int = 42
    ):
        """
        Initialize PSO optimizer.

        Args:
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            w_decay: Inertia weight decay per iteration
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.seed = seed

    def _initialize_particles(
        self,
        bounds: List[Tuple[float, float]],
        param_types: List[str]
    ) -> List[Particle]:
        """Initialize particle swarm."""
        np.random.seed(self.seed)
        n_dims = len(bounds)
        particles = []

        for i in range(self.n_particles):
            # Random position within bounds
            position = np.array([
                np.random.uniform(low, high) for low, high in bounds
            ])

            # Apply parameter types
            position = self._apply_param_types(position, param_types, bounds)

            # Random initial velocity
            velocity = np.array([
                np.random.uniform(-(high-low)*0.1, (high-low)*0.1)
                for low, high in bounds
            ])

            particles.append(Particle(position=position, velocity=velocity))

        return particles

    def _apply_param_types(
        self,
        position: np.ndarray,
        param_types: List[str],
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Apply parameter type constraints."""
        if param_types is None:
            return position

        for i, ptype in enumerate(param_types):
            if ptype == 'integer':
                position[i] = np.round(position[i])
            elif ptype == 'log':
                # Position is already in linear space, just ensure bounds
                position[i] = np.clip(position[i], bounds[i][0], bounds[i][1])

        return position

    def optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        param_types: List[str] = None,
        callback: Callable = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run PSO optimization.

        Args:
            objective: Function to minimize (takes position array, returns scalar)
            bounds: List of (min, max) tuples for each parameter
            param_types: List of parameter types
            callback: Optional callback function(iteration, best_fitness)

        Returns:
            Tuple of (best_position, best_fitness, history)
        """
        logger = get_logger()
        logger.info(f"Starting PSO: {self.n_particles} particles, {self.n_iterations} iterations")

        # Initialize
        particles = self._initialize_particles(bounds, param_types)
        global_best_position = None
        global_best_fitness = float('inf')

        history = {
            'iterations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'positions': []
        }

        w = self.w

        # Main loop
        for iteration in range(self.n_iterations):
            iteration_fitness = []

            for particle in particles:
                # Evaluate fitness
                fitness = objective(particle.position)
                particle.current_fitness = fitness
                iteration_fitness.append(fitness)

                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()

            # Update velocities and positions
            for particle in particles:
                r1, r2 = np.random.random(2)

                # Velocity update
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social

                # Position update
                particle.position = particle.position + particle.velocity

                # Apply bounds
                for i, (low, high) in enumerate(bounds):
                    particle.position[i] = np.clip(particle.position[i], low, high)

                # Apply parameter types
                particle.position = self._apply_param_types(particle.position, param_types, bounds)

            # Decay inertia weight
            w *= self.w_decay

            # Record history
            history['iterations'].append(iteration)
            history['best_fitness'].append(global_best_fitness)
            history['mean_fitness'].append(np.mean(iteration_fitness))
            history['positions'].append(global_best_position.copy())

            # Callback
            if callback is not None:
                callback(iteration, global_best_fitness)

            # Log progress
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                logger.info(f"  Iteration {iteration}: Best fitness = {global_best_fitness:.6f}")

        logger.info(f"PSO complete. Best fitness: {global_best_fitness:.6f}")

        return global_best_position, global_best_fitness, history


class GWO(SwarmOptimizer):
    """
    Grey Wolf Optimizer implementation.

    Based on the hunting behavior of grey wolves.
    """

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 30,
        seed: int = 42
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.seed = seed

    def optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        param_types: List[str] = None,
        callback: Callable = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run GWO optimization.

        Args:
            objective: Function to minimize
            bounds: List of (min, max) tuples
            param_types: List of parameter types
            callback: Optional callback function

        Returns:
            Tuple of (best_position, best_fitness, history)
        """
        logger = get_logger()
        logger.info(f"Starting GWO: {self.n_particles} wolves, {self.n_iterations} iterations")

        np.random.seed(self.seed)
        n_dims = len(bounds)

        # Initialize population
        positions = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(self.n_particles)
        ])

        # Apply parameter types
        for i in range(self.n_particles):
            if param_types:
                for j, ptype in enumerate(param_types):
                    if ptype == 'integer':
                        positions[i, j] = np.round(positions[i, j])

        # Initialize alpha, beta, delta (top 3 wolves)
        alpha_pos = np.zeros(n_dims)
        alpha_score = float('inf')
        beta_pos = np.zeros(n_dims)
        beta_score = float('inf')
        delta_pos = np.zeros(n_dims)
        delta_score = float('inf')

        history = {
            'iterations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'positions': []
        }

        # Main loop
        for iteration in range(self.n_iterations):
            # Evaluate fitness and update hierarchy
            fitness_values = []

            for i in range(self.n_particles):
                fitness = objective(positions[i])
                fitness_values.append(fitness)

                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[i].copy()

            # a decreases linearly from 2 to 0
            a = 2 - iteration * (2 / self.n_iterations)

            # Update positions
            for i in range(self.n_particles):
                for j in range(n_dims):
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3

                # Apply bounds
                for j, (low, high) in enumerate(bounds):
                    positions[i, j] = np.clip(positions[i, j], low, high)

                # Apply parameter types
                if param_types:
                    for j, ptype in enumerate(param_types):
                        if ptype == 'integer':
                            positions[i, j] = np.round(positions[i, j])

            # Record history
            history['iterations'].append(iteration)
            history['best_fitness'].append(alpha_score)
            history['mean_fitness'].append(np.mean(fitness_values))
            history['positions'].append(alpha_pos.copy())

            if callback:
                callback(iteration, alpha_score)

            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                logger.info(f"  Iteration {iteration}: Best fitness = {alpha_score:.6f}")

        logger.info(f"GWO complete. Best fitness: {alpha_score:.6f}")

        return alpha_pos, alpha_score, history


def get_optimizer(name: str, **kwargs) -> SwarmOptimizer:
    """
    Get optimizer by name.

    Args:
        name: Optimizer name ('pso' or 'gwo')
        **kwargs: Optimizer parameters

    Returns:
        Optimizer instance
    """
    optimizers = {
        'pso': PSO,
        'gwo': GWO
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name](**kwargs)
