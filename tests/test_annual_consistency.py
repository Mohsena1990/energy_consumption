"""
Tests for annual consistency safeguards.
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safeguards import (
    aggregate_quarterly_to_annual,
    check_annual_consistency,
    create_annual_baseline,
    compare_with_annual_baseline
)
from src.core.utils import inverse_log_transform


class TestAggregation:
    """Tests for quarterly to annual aggregation."""

    @pytest.fixture
    def quarterly_predictions(self):
        """Create sample quarterly predictions."""
        dates = pd.date_range(start='2010-01-01', periods=40, freq='QS')

        np.random.seed(42)
        actual = np.random.randn(40) + 10  # In log space
        predicted = actual + np.random.randn(40) * 0.1  # Small error

        return pd.DataFrame({
            'date': dates,
            'actual': actual,
            'predicted': predicted
        })

    def test_aggregation_basic(self, quarterly_predictions):
        """Test basic aggregation to annual."""
        annual = aggregate_quarterly_to_annual(quarterly_predictions, transform='log')

        assert 'year' in annual.columns
        assert 'actual_annual' in annual.columns
        assert 'predicted_annual' in annual.columns

        # Should have fewer rows than quarterly
        assert len(annual) < len(quarterly_predictions)

    def test_aggregation_years(self, quarterly_predictions):
        """Test that aggregation produces correct years."""
        annual = aggregate_quarterly_to_annual(quarterly_predictions, transform='log')

        expected_years = quarterly_predictions['date'].dt.year.unique()
        actual_years = annual['year'].unique()

        np.testing.assert_array_equal(
            sorted(expected_years),
            sorted(actual_years)
        )

    def test_aggregation_log_transform(self):
        """Test aggregation with log transform."""
        # Create data where we know the expected annual values
        dates = pd.date_range(start='2020-01-01', periods=4, freq='QS')

        # Values in log space
        log_values = np.array([1.0, 1.0, 1.0, 1.0])  # log(e) for each quarter

        df = pd.DataFrame({
            'date': dates,
            'actual': log_values,
            'predicted': log_values
        })

        annual = aggregate_quarterly_to_annual(df, transform='log')

        # exp(1) + exp(1) + exp(1) + exp(1) = 4 * e ≈ 10.87
        expected_annual = 4 * np.exp(1.0)
        actual_annual = annual['actual_annual'].iloc[0]

        np.testing.assert_almost_equal(actual_annual, expected_annual, decimal=2)


class TestConsistencyCheck:
    """Tests for annual consistency checking."""

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions (no error)."""
        dates = pd.date_range(start='2015-01-01', periods=20, freq='QS')
        values = np.random.randn(20) + 10

        return pd.DataFrame({
            'date': dates,
            'actual': values,
            'predicted': values  # Perfect match
        })

    @pytest.fixture
    def noisy_predictions(self):
        """Create predictions with noise."""
        dates = pd.date_range(start='2015-01-01', periods=20, freq='QS')
        actual = np.random.randn(20) + 10
        predicted = actual + np.random.randn(20) * 0.5  # Add noise

        return pd.DataFrame({
            'date': dates,
            'actual': actual,
            'predicted': predicted
        })

    def test_consistency_perfect(self, perfect_predictions):
        """Test consistency with perfect predictions."""
        result = check_annual_consistency(perfect_predictions, transform='none')

        assert result['annual_mae'] < 0.01, "MAE should be ~0 for perfect predictions"
        assert result['annual_mape'] < 0.01, "MAPE should be ~0 for perfect predictions"
        assert result['consistency_score'] > 0.99, "Consistency should be ~1"

    def test_consistency_noisy(self, noisy_predictions):
        """Test consistency with noisy predictions."""
        result = check_annual_consistency(noisy_predictions, transform='none')

        assert result['annual_mae'] > 0, "MAE should be > 0 for noisy predictions"
        assert 'annual_mape' in result
        assert 'consistency_score' in result
        assert 0 <= result['consistency_score'] <= 1

    def test_consistency_output_structure(self, noisy_predictions):
        """Test output structure of consistency check."""
        result = check_annual_consistency(noisy_predictions, transform='none')

        assert 'annual_mae' in result
        assert 'annual_mape' in result
        assert 'annual_bias' in result
        assert 'n_years' in result
        assert 'year_errors' in result


class TestBaselines:
    """Tests for annual baseline models."""

    @pytest.fixture
    def annual_series(self):
        """Create annual time series."""
        years = range(2000, 2020)
        values = 100 + np.arange(20) * 2 + np.random.randn(20) * 5

        return pd.Series(values, index=years)

    def test_naive_baseline(self, annual_series):
        """Test naive baseline (last year's value)."""
        predictions, info = create_annual_baseline(annual_series, method='naive')

        assert len(predictions) == len(annual_series)
        assert info['method'] == 'naive'

        # Naive baseline: prediction[t] = actual[t-1]
        for i in range(1, len(predictions)):
            np.testing.assert_almost_equal(
                predictions[i],
                annual_series.iloc[i-1]
            )

    def test_mean_baseline(self, annual_series):
        """Test mean baseline (expanding mean)."""
        predictions, info = create_annual_baseline(annual_series, method='mean')

        assert len(predictions) == len(annual_series)
        assert info['method'] == 'mean'

    def test_linear_baseline(self, annual_series):
        """Test linear trend baseline."""
        predictions, info = create_annual_baseline(annual_series, method='linear')

        assert len(predictions) == len(annual_series)
        assert info['method'] == 'linear'


class TestBaselineComparison:
    """Tests for comparing models with baselines."""

    def test_comparison_structure(self):
        """Test the structure of baseline comparison results."""
        dates = pd.date_range(start='2015-01-01', periods=20, freq='QS')
        actual = np.random.randn(20) + 10
        predicted = actual + np.random.randn(20) * 0.1

        predictions_df = pd.DataFrame({
            'date': dates,
            'actual': actual,
            'predicted': predicted
        })

        # Create annual true values
        y_annual = pd.Series(
            np.random.randn(5) + 100,
            index=range(2015, 2020)
        )

        result = compare_with_annual_baseline(
            predictions_df, y_annual,
            transform='none',
            baseline_methods=['naive', 'mean']
        )

        if 'error' not in result:
            assert 'model' in result
            assert 'baselines' in result
            assert 'naive' in result['baselines'] or len(result['baselines']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
