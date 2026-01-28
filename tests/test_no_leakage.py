"""
Tests for ensuring no data leakage in the forecasting pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.splits import (
    create_walk_forward_splits, generate_cv_folds, validate_no_leakage, CVPlan
)
from src.core import SplitConfig
from src.features import create_lag_features


class TestNoLeakage:
    """Tests for no data leakage."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2000-01-01', periods=100, freq='QS')
        np.random.seed(42)

        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        }, index=dates)

        y = pd.Series(np.random.randn(100), index=dates, name='target')

        return X, y

    @pytest.fixture
    def cv_config(self):
        """Create CV configuration."""
        return SplitConfig(
            method='walk_forward',
            min_train_size=40,
            test_size=4,
            horizons=[1, 2, 4]
        )

    def test_train_test_temporal_separation(self, sample_data, cv_config):
        """Test that training data always precedes test data."""
        X, y = sample_data
        cv_plan = create_walk_forward_splits(X, y, cv_config)

        for fold in cv_plan.folds:
            # All training indices should be less than test indices
            assert fold.train_indices.max() < fold.test_indices.min(), \
                f"Fold {fold.fold_id}: Training data not before test data"

            # Check dates
            train_max_date = X.index[fold.train_indices[-1]]
            test_min_date = X.index[fold.test_indices[0]]

            assert train_max_date < test_min_date, \
                f"Fold {fold.fold_id}: Training date {train_max_date} not before test date {test_min_date}"

    def test_no_overlapping_indices(self, sample_data, cv_config):
        """Test that train and test sets don't overlap."""
        X, y = sample_data
        cv_plan = create_walk_forward_splits(X, y, cv_config)

        for fold in cv_plan.folds:
            train_set = set(fold.train_indices)
            test_set = set(fold.test_indices)

            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, \
                f"Fold {fold.fold_id}: Found overlapping indices: {overlap}"

    def test_validate_no_leakage_function(self, sample_data, cv_config):
        """Test the validate_no_leakage utility function."""
        X, y = sample_data
        cv_plan = create_walk_forward_splits(X, y, cv_config)

        for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
            assert validate_no_leakage(X_train, X_test, y_train, y_test), \
                f"Fold {fold.fold_id}: Leakage detected by validate_no_leakage"

    def test_lag_features_no_future_leakage(self):
        """Test that lag features don't leak future information."""
        dates = pd.date_range(start='2000-01-01', periods=20, freq='QS')
        values = np.arange(20)

        df = pd.DataFrame({'value': values}, index=dates)

        # Create lag features
        lag_df = create_lag_features(df, ['value'], lags=[1, 2, 3])

        # For each row, lag values should only come from previous rows
        for i in range(3, len(df)):
            current_date = df.index[i]

            # lag1 should be value from i-1
            assert lag_df.iloc[i]['value_lag1'] == values[i-1], \
                f"Lag1 at index {i} is incorrect"

            # lag2 should be value from i-2
            assert lag_df.iloc[i]['value_lag2'] == values[i-2], \
                f"Lag2 at index {i} is incorrect"

            # lag3 should be value from i-3
            assert lag_df.iloc[i]['value_lag3'] == values[i-3], \
                f"Lag3 at index {i} is incorrect"

    def test_scaler_fit_only_on_train(self, sample_data, cv_config):
        """Test that scalers are fit only on training data."""
        from sklearn.preprocessing import StandardScaler

        X, y = sample_data
        cv_plan = create_walk_forward_splits(X, y, cv_config)

        for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
            # Fit scaler on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # The scaler should have mean/std from training data
            train_mean = X_train.mean().values
            train_std = X_train.std().values

            np.testing.assert_array_almost_equal(
                scaler.mean_, train_mean, decimal=5,
                err_msg=f"Fold {fold.fold_id}: Scaler mean doesn't match training mean"
            )

            # Transform test data (without fitting)
            X_test_scaled = scaler.transform(X_test)

            # Verify test data was transformed using training statistics
            # (test scaled values may not have mean=0, std=1)
            # This is expected behavior - no leakage
            assert X_test_scaled.shape == X_test.shape


class TestWalkForwardIntegrity:
    """Tests for walk-forward CV integrity."""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2000-01-01', periods=80, freq='QS')
        X = pd.DataFrame({'feature': np.arange(80)}, index=dates)
        y = pd.Series(np.arange(80), index=dates)
        return X, y

    def test_expanding_window(self, sample_data):
        """Test that training window expands over folds."""
        X, y = sample_data

        config = SplitConfig(
            min_train_size=30,
            test_size=4,
            horizons=[1]
        )

        cv_plan = create_walk_forward_splits(X, y, config)

        train_sizes = [len(f.train_indices) for f in cv_plan.folds]

        # Training size should be non-decreasing
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], \
                f"Training size decreased from fold {i-1} to {i}"

    def test_horizon_offset(self, sample_data):
        """Test that forecast horizons are correctly applied."""
        X, y = sample_data

        config = SplitConfig(
            min_train_size=30,
            test_size=4,
            horizons=[1, 2, 4]
        )

        cv_plan = create_walk_forward_splits(X, y, config)

        for fold in cv_plan.folds:
            train_end_idx = fold.train_indices[-1]
            test_start_idx = fold.test_indices[0]

            # Gap should be at least the horizon
            gap = test_start_idx - train_end_idx
            assert gap >= fold.horizon, \
                f"Fold {fold.fold_id}: Gap {gap} < horizon {fold.horizon}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
