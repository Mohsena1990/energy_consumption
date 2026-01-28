# CO2 Forecasting Framework

A modular Python framework for quarterly CO2 emissions forecasting with accounting-consistent validation, SHAP-based feature selection evaluation, and MCDA decision-making.

## Overview

This framework implements a comprehensive pipeline for CO2 forecasting that includes:

- **Three Feature Selection Strategies**: Linear (VIF + Ridge/ElasticNet), Nonlinear (RF/LightGBM/CatBoost), Consensus
- **SHAP-based FS Evaluation**: Evaluate feature selection options using SHAP values across walk-forward CV
- **MCDA Decision Making**: VIKOR and TOPSIS methods for selecting best FS option and final model
- **Walk-forward Cross-Validation**: No data leakage, proper temporal validation
- **Swarm Optimization**: PSO and GWO for hyperparameter tuning
- **Annual Consistency Safeguards**: Verify quarterly predictions aggregate correctly to annual totals
- **High-Resolution Visualization**: Publication-quality plots at 300 DPI

## Models

- Ridge Regression
- Random Forest
- CatBoost
- LightGBM
- LSTM (PyTorch)

## Installation

```bash
# Clone or navigate to the repository
cd "c:\Users\Ilani\OneDrive\Desktop\Shahla\New folder"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── configs/                    # Configuration files
│   └── default_config.yaml
├── data/                       # Data directory
│   └── processed/             # Processed data (generated)
├── outputs/                    # Output directory
│   └── runs/                  # Run outputs
│       └── run_YYYYMMDD_HHMM/
│           ├── configs_snapshot/
│           ├── figures/
│           ├── logs/
│           ├── metrics/
│           ├── models/
│           ├── predictions/
│           └── tables/
├── scripts/                    # Entry-point scripts
│   ├── 00_make_dataset.py
│   ├── 01_run_fs.py
│   ├── 02_eval_fs_shap_mcda.py
│   ├── 03_optimize_models.py
│   ├── 04_evaluate_and_safeguards.py
│   ├── 05_select_best_model.py
│   └── 06_interpret_champion.py
├── src/                        # Source modules
│   ├── core/                  # Config, logging, utilities
│   ├── data_io/               # Data loading, schema
│   ├── quality/               # Data quality checks
│   ├── features/              # Feature engineering
│   ├── splits/                # Walk-forward CV
│   ├── fs/                    # Feature selection
│   ├── models/                # Forecasting models
│   ├── optimization/          # PSO/GWO optimization
│   ├── evaluation/            # Metrics
│   ├── safeguards/            # Annual consistency
│   ├── decision/              # MCDA (VIKOR/TOPSIS)
│   ├── interpretability/      # SHAP analysis
│   └── reporting/             # Plotting
├── tests/                      # Test files
├── requirements.txt
└── README.md
```

## Usage

### Quick Start (Run Full Pipeline)

```bash
# Step 0: Load and prepare data
python scripts/00_make_dataset.py --input "data 1999-2025Q1.xlsx"

# Step 1: Run all feature selection methods
python scripts/01_run_fs.py

# Step 2: Evaluate FS options with SHAP and select best using MCDA
python scripts/02_eval_fs_shap_mcda.py

# Step 3: Optimize all models using selected features
python scripts/03_optimize_models.py

# Step 4: Evaluate models and apply annual consistency safeguards
python scripts/04_evaluate_and_safeguards.py

# Step 5: Select best model using Pareto + MCDA
python scripts/05_select_best_model.py

# Step 6: Generate interpretability report for champion model
python scripts/06_interpret_champion.py
```

### Step-by-Step Guide

#### 1. Data Preparation (Script 00)

Load raw Excel data, perform quality checks, and create feature-engineered dataset.

```bash
python scripts/00_make_dataset.py --input "data 1999-2025Q1.xlsx"
```

**Outputs:**
- `data/processed/df_clean.parquet` - Cleaned data
- `data/processed/X_full.parquet` - Feature matrix
- `data/processed/y.parquet` - Target variable
- `data/processed/cv_plan.pkl` - Walk-forward CV plan
- Quality reports and feature dictionary

#### 2. Feature Selection (Script 01)

Run three FS strategies: linear, nonlinear, and consensus.

```bash
python scripts/01_run_fs.py
```

**Outputs:**
- FS scores for each method
- Selected feature lists

#### 3. FS Evaluation with SHAP + MCDA (Script 02)

Evaluate FS options using SHAP-based metrics and select the best.

```bash
python scripts/02_eval_fs_shap_mcda.py
```

**Evaluation Criteria:**
- C1: Accuracy (weighted MAE)
- C2: Stability (std of errors)
- C3: SHAP concentration (top-K share)
- C4: SHAP stability (rank correlation)
- C5: Parsimony (number of features)

**Outputs:**
- `fs_evaluation_matrix.csv`
- `fs_mcda_ranking.csv`
- `selected_feature_set.json`

#### 4. Model Optimization (Script 03)

Train and optimize all models using PSO/GWO.

```bash
python scripts/03_optimize_models.py
```

**Outputs:**
- Best parameters for each model
- Optimization history plots
- Trained model files

#### 5. Evaluation and Safeguards (Script 04)

Evaluate models with quarterly metrics and annual consistency checks.

```bash
python scripts/04_evaluate_and_safeguards.py
```

**Safeguards:**
- Aggregate quarterly predictions to annual totals
- Compare with observed annual values
- Benchmark against simple annual baselines

**Outputs:**
- `quarterly_metrics.csv`
- `annual_consistency.csv`
- Prediction plots

#### 6. Final Model Selection (Script 05)

Select champion model using Pareto front and MCDA.

```bash
python scripts/05_select_best_model.py
```

**Outputs:**
- `pareto_front.csv`
- `mcda_model_ranking.csv`
- `champion_pipeline.json`

#### 7. Champion Interpretation (Script 06)

Generate SHAP explanations and regime analysis.

```bash
python scripts/06_interpret_champion.py
```

**Outputs:**
- SHAP summary plot
- Regime comparison (pre/post COVID)
- Seasonal leverage analysis
- Top drivers table

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
# Key settings
data:
  target_transform: "log"  # or "delta_log"

splits:
  min_train_size: 40
  horizons: [1, 2, 4]
  horizon_weights:
    1: 0.5
    2: 0.3
    4: 0.2

optimization:
  optimizer: "pso"  # or "gwo"
  n_particles: 20
  n_iterations: 30

mcda:
  method: "vikor"  # or "topsis"
```

## Custom Configuration

```bash
# Use custom config
python scripts/00_make_dataset.py --config configs/my_config.yaml

# Continue with same run
python scripts/01_run_fs.py --run-id run_20240101_1200
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_no_leakage.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Output Figures

The framework generates high-resolution (300 DPI) figures:

- **Optimization History**: Convergence plots for PSO/GWO
- **Pareto Front**: Multi-objective visualization
- **MCDA Ranking**: VIKOR/TOPSIS ranking comparison
- **Predictions vs Actual**: Time series comparison
- **Annual Consistency**: Quarterly-to-annual aggregation
- **SHAP Summary**: Feature importance visualization
- **Regime Comparison**: Pre/post COVID feature importance
- **Seasonal Leverage**: Quarterly sensitivity analysis

## Key Constraints

1. **No Random Splits**: Only walk-forward/expanding window CV
2. **No Leakage**: Scalers fit on training fold only; lags computed without future data
3. **LSTM Constraints**: Small model (limited lookback, dropout, early stopping)
4. **Annual Consistency**: Quarterly predictions must aggregate to sensible annual totals

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn, lightgbm, catboost
- torch (for LSTM)
- shap, matplotlib
- See `requirements.txt` for full list

## License

MIT License
