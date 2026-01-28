"""
Configuration management for CO2 forecasting framework.
"""
import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class DataConfig:
    """Data loading configuration."""
    input_path: str = "data/raw/data_1999-2025Q1.xlsx"
    date_column: str = "Quarter"  # or "Date"
    target_column: str = "CO2e"
    target_transform: str = "log"  # "log" or "delta_log"
    sheet_name: Optional[str] = None


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lag_features: List[str] = field(default_factory=lambda: ["CO2e"])
    lag_orders: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    seasonality_type: str = "dummies"  # "dummies" or "sincos"
    include_covid_dummy: bool = True
    covid_start: str = "2020Q1"
    covid_end: str = "2021Q4"
    include_energy_crisis: bool = True
    energy_crisis_start: str = "2022Q1"
    energy_crisis_end: str = "2023Q4"


@dataclass
class SplitConfig:
    """Cross-validation split configuration."""
    method: str = "walk_forward"  # "walk_forward" or "expanding"
    min_train_size: int = 40  # minimum quarters for training
    test_size: int = 4  # quarters per test fold
    horizons: List[int] = field(default_factory=lambda: [1, 2, 4])
    horizon_weights: Dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.3, 4: 0.2})


@dataclass
class FSConfig:
    """Feature selection configuration."""
    vif_threshold: float = 10.0
    stability_threshold: float = 0.6  # 60% of folds
    vote_threshold: int = 2  # minimum methods agreeing
    top_k_features: int = 5  # for SHAP concentration
    evaluator_model: str = "lightgbm"  # or "catboost"


@dataclass
class OptimizationConfig:
    """Swarm optimization configuration."""
    optimizer: str = "pso"  # "pso" or "gwo"
    n_particles: int = 20
    n_iterations: int = 30
    seed: int = 42
    annual_penalty_threshold: float = 0.05  # 5% MAPE threshold
    annual_penalty_weight: float = 0.1


@dataclass
class ModelConfig:
    """Model training configuration."""
    models: List[str] = field(default_factory=lambda: [
        "ridge", "random_forest", "catboost", "lightgbm", "lstm"
    ])
    lstm_lookback_options: List[int] = field(default_factory=lambda: [4, 8, 12])
    lstm_max_epochs: int = 200
    lstm_patience: int = 20
    lstm_dropout: float = 0.2


@dataclass
class MCDAConfig:
    """MCDA decision maker configuration."""
    method: str = "vikor"  # "vikor" or "topsis"
    use_pareto_filter: bool = True
    # Weights for FS evaluation criteria
    fs_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.30,
        "stability": 0.20,
        "shap_concentration": 0.15,
        "shap_stability": 0.20,
        "parsimony": 0.15
    })
    # Weights for final model selection
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "quarterly_mae": 0.35,
        "stability": 0.20,
        "annual_consistency": 0.25,
        "interpretability": 0.10,
        "parsimony": 0.10
    })
    vikor_v: float = 0.5  # VIKOR compromise parameter


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "outputs"
    figure_dpi: int = 300
    figure_format: str = "png"
    save_models: bool = True
    save_predictions: bool = True


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    fs: FSConfig = field(default_factory=FSConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcda: MCDAConfig = field(default_factory=MCDAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    run_id: Optional[str] = None

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M")

    @property
    def run_dir(self) -> Path:
        return Path(self.output.base_dir) / "runs" / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.run_dir / "configs_snapshot" / "config.yaml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            data=DataConfig(**data.get('data', {})),
            features=FeatureConfig(**data.get('features', {})),
            splits=SplitConfig(**data.get('splits', {})),
            fs=FSConfig(**data.get('fs', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            model=ModelConfig(**data.get('model', {})),
            mcda=MCDAConfig(**data.get('mcda', {})),
            output=OutputConfig(**data.get('output', {})),
            seed=data.get('seed', 42),
            run_id=data.get('run_id')
        )


def create_run_directories(config: Config) -> Dict[str, Path]:
    """Create all output directories for a run."""
    run_dir = config.run_dir
    dirs = {
        'root': run_dir,
        'logs': run_dir / 'logs',
        'tables': run_dir / 'tables',
        'metrics': run_dir / 'metrics',
        'predictions': run_dir / 'predictions',
        'figures': run_dir / 'figures',
        'models': run_dir / 'models',
        'configs_snapshot': run_dir / 'configs_snapshot'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_latest_run_id(base_dir: str = "outputs") -> Optional[str]:
    """Find the most recent run_id by sorting run directories."""
    runs_dir = Path(base_dir) / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0].name
    return None


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
