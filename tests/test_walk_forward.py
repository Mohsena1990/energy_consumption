"""
Tests for walk-forward cross-validation.
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.splits import (
    create_walk_forward_splits, CVPlan, CVFold, generate_cv_folds
)
from src.core import SplitConfig


class TestWalkForwardCV:
    """Tests for walk-forward cross-validation."""

    @pytest.fixture
    def quarterly_data(self):
        """Create quarterly time series data."""
        dates = pd.date_range(start='1999-01-01', periods=100, freq='QS')
        np.random.seed(42)

        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(100)
            for i in range(5)
        }, index=dates)

        y = pd.Series(
            np.cumsum(np.random.randn(100)) + 100,
            index=dates,
            name='target'
        )

        return X, y

    @pytest.fixture
    def config(self):
        """Default CV configuration."""
        return SplitConfig(
            method='walk_forward',
            min_train_size=40,
            test_size=4,
            horizons=[1, 2, 4]
        )

    def test_cv_plan_creation(self, quarterly_data, config):
        """Test that CV plan is created correctly."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        assert isinstance(cv_plan, CVPlan)
        assert cv_plan.n_folds > 0

    def test_fold_count_by_horizon(self, quarterly_data, config):
        """Test that folds are created for each horizon."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        for horizon in config.horizons:
            folds_for_horizon = cv_plan.get_folds_by_horizon(horizon)
            assert len(folds_for_horizon) > 0, f"No folds for horizon {horizon}"

    def test_minimum_train_size(self, quarterly_data, config):
        """Test that training sets meet minimum size requirement."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        for fold in cv_plan.folds:
            assert len(fold.train_indices) >= config.min_train_size, \
                f"Fold {fold.fold_id}: Train size {len(fold.train_indices)} < min {config.min_train_size}"

    def test_test_size(self, quarterly_data, config):
        """Test that test sets have expected size."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        for fold in cv_plan.folds:
            assert len(fold.test_indices) <= config.test_size, \
                f"Fold {fold.fold_id}: Test size {len(fold.test_indices)} > max {config.test_size}"

    def test_temporal_ordering(self, quarterly_data, config):
        """Test that dates are properly ordered."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        for fold in cv_plan.folds:
            assert fold.train_start <= fold.train_end, "Train start after train end"
            assert fold.test_start <= fold.test_end, "Test start after test end"
            assert fold.train_end < fold.test_start, "Train end not before test start"

    def test_generate_cv_folds_data_shapes(self, quarterly_data, config):
        """Test that generated fold data has correct shapes."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        for X_train, y_train, X_test, y_test, fold in generate_cv_folds(X, y, cv_plan):
            assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
            assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
            assert len(X_train) == len(fold.train_indices), "X_train length != train_indices"
            assert len(X_test) == len(fold.test_indices), "X_test length != test_indices"

    def test_cv_plan_to_dataframe(self, quarterly_data, config):
        """Test conversion to DataFrame."""
        X, y = quarterly_data
        cv_plan = create_walk_forward_splits(X, y, config)

        df = cv_plan.to_dataframe()

        assert 'fold_id' in df.columns
        assert 'horizon' in df.columns
        assert 'train_start' in df.columns
        assert 'train_end' in df.columns
        assert 'test_start' in df.columns
        assert 'test_end' in df.columns
        assert len(df) == cv_plan.n_folds


class TestCVFold:
    """Tests for CVFold dataclass."""

    def test_fold_creation(self):
        """Test CVFold creation."""
        fold = CVFold(
            fold_id=0,
            train_start=pd.Timestamp('2000-01-01'),
            train_end=pd.Timestamp('2010-01-01'),
            test_start=pd.Timestamp('2010-04-01'),
            test_end=pd.Timestamp('2011-01-01'),
            train_indices=np.arange(40),
            test_indices=np.arange(40, 44),
            horizon=1
        )

        assert fold.fold_id == 0
        assert fold.horizon == 1
        assert len(fold.train_indices) == 40
        assert len(fold.test_indices) == 4


class TestEdgeCases:
    """Tests for edge cases in walk-forward CV."""

    def test_small_dataset(self):
        """Test with small dataset."""
        dates = pd.date_range(start='2000-01-01', periods=50, freq='QS')
        X = pd.DataFrame({'feature': np.random.randn(50)}, index=dates)
        y = pd.Series(np.random.randn(50), index=dates)

        config = SplitConfig(
            min_train_size=40,
            test_size=4,
            horizons=[1]
        )

        cv_plan = create_walk_forward_splits(X, y, config)

        # Should still create some folds
        assert cv_plan.n_folds > 0

    def test_single_horizon(self):
        """Test with single horizon."""
        dates = pd.date_range(start='2000-01-01', periods=80, freq='QS')
        X = pd.DataFrame({'feature': np.random.randn(80)}, index=dates)
        y = pd.Series(np.random.randn(80), index=dates)

        config = SplitConfig(
            min_train_size=40,
            test_size=4,
            horizons=[1]  # Single horizon
        )

        cv_plan = create_walk_forward_splits(X, y, config)

        # All folds should have horizon=1
        for fold in cv_plan.folds:
            assert fold.horizon == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
