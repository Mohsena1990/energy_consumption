"""
Safeguards module for CO2 forecasting framework.
Annual consistency checks and baseline comparisons.
"""
from .annual_consistency import (
    aggregate_quarterly_to_annual,
    check_annual_consistency,
    create_annual_baseline,
    compare_with_annual_baseline
)

__all__ = [
    'aggregate_quarterly_to_annual',
    'check_annual_consistency',
    'create_annual_baseline',
    'compare_with_annual_baseline'
]
