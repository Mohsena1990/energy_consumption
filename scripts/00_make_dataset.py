#!/usr/bin/env python
"""
Script 00: Make Dataset
=======================
Load raw data, perform quality checks, and create feature-engineered dataset.

Usage:
    python scripts/00_make_dataset.py [--config CONFIG_PATH] [--input INPUT_PATH]

Outputs:
    - data/processed/df_clean.parquet
    - data/processed/X_full.parquet
    - data/processed/y.parquet
    - outputs/runs/<run_id>/tables/data_quality_report.csv
    - outputs/runs/<run_id>/tables/feature_dictionary.csv
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    Config, create_run_directories, setup_logging, get_logger,
    set_seed, save_json_numpy
)
from src.data_io import (
    load_and_prepare_data, save_processed_data, create_default_schema
)
from src.quality import generate_quality_report, clean_data
from src.features import engineer_features, create_feature_dictionary
from src.splits import create_walk_forward_splits, save_cv_plan


def parse_args():
    parser = argparse.ArgumentParser(description='Make dataset for CO2 forecasting')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input Excel file')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    # Override input path if provided
    if args.input:
        config.data.input_path = args.input

    # Set seed
    set_seed(config.seed)

    # Create run directories
    dirs = create_run_directories(config)

    # Setup logging
    logger = setup_logging(
        log_dir=dirs['logs'],
        run_id=config.run_id
    )

    logger.info("=" * 60)
    logger.info("Script 00: Make Dataset")
    logger.info("=" * 60)
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Input: {config.data.input_path}")

    # Save config
    config.save(dirs['configs_snapshot'] / 'config.yaml')

    # =========================================
    # Step 1: Load Raw Data
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 1: Loading raw data...")

    schema = create_default_schema()
    df_raw, metadata = load_and_prepare_data(config, schema)

    save_json_numpy(metadata, dirs['tables'] / 'data_metadata.json')
    logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # =========================================
    # Step 2: Data Quality Audit
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 2: Data quality audit...")

    quality_report = generate_quality_report(df_raw, dirs['tables'])
    logger.info(f"Quality report saved to {dirs['tables']}")

    # =========================================
    # Step 3: Clean Data
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 3: Cleaning data...")

    df_clean, cleaning_log = clean_data(df_raw)
    save_json_numpy(cleaning_log, dirs['tables'] / 'cleaning_log.json')

    # Save clean data
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_processed_data(df_clean, processed_dir, 'df_clean')
    logger.info(f"Cleaned data: {len(df_clean)} rows, {len(df_clean.columns)} columns")

    # =========================================
    # Step 4: Feature Engineering
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 4: Feature engineering...")

    X, y, feature_metadata = engineer_features(df_clean, config)

    # Drop rows with NaN (from lag features)
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Features: {len(X.columns)} columns, {len(X)} valid rows")

    # Save feature data
    save_processed_data(X, processed_dir, 'X_full')
    y.to_frame('target').to_parquet(processed_dir / 'y.parquet')

    # Create feature dictionary
    feature_dict = create_feature_dictionary(
        X, feature_metadata,
        output_path=dirs['tables'] / 'feature_dictionary.csv'
    )
    save_json_numpy(feature_metadata, dirs['tables'] / 'feature_metadata.json')

    # =========================================
    # Step 5: Create CV Splits
    # =========================================
    logger.info("-" * 40)
    logger.info("Step 5: Creating walk-forward CV splits...")

    cv_plan = create_walk_forward_splits(X, y, config.splits)
    save_cv_plan(cv_plan, processed_dir / 'cv_plan')

    logger.info(f"Created {cv_plan.n_folds} CV folds")

    # Save CV plan summary
    cv_plan.to_dataframe().to_csv(dirs['tables'] / 'cv_plan.csv', index=False)

    # =========================================
    # Summary
    # =========================================
    logger.info("=" * 60)
    logger.info("Dataset creation complete!")
    logger.info(f"  Raw data: {len(df_raw)} rows")
    logger.info(f"  Clean data: {len(df_clean)} rows")
    logger.info(f"  Features: {len(X.columns)} columns")
    logger.info(f"  Valid samples: {len(X)}")
    logger.info(f"  CV folds: {cv_plan.n_folds}")
    logger.info(f"  Outputs saved to: {dirs['root']}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
