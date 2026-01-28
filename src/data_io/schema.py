"""
Schema mapping for CO2 forecasting data.
Provides flexible column name mapping to handle different data formats.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class DataSchema:
    """
    Schema definition for CO2 forecasting data.
    Maps logical column names to actual column names in the dataset.
    """
    # Date/Time column (can be "Quarter", "Date", "Period", etc.)
    date_column: str = "Quarter"

    # Target variable (CO2 emissions)
    target_column: str = "CO2e"

    # Economic indicators
    economic_columns: List[str] = field(default_factory=lambda: [
        "GDP", "GDP_growth", "Industrial_Production", "Population",
        "GDP_per_capita", "Consumption", "Investment", "Exports", "Imports"
    ])

    # Energy indicators
    energy_columns: List[str] = field(default_factory=lambda: [
        "Energy_Consumption", "Coal_Consumption", "Oil_Consumption",
        "Gas_Consumption", "Renewable_Share", "Electricity_Generation",
        "Energy_Intensity"
    ])

    # Climate indicators
    climate_columns: List[str] = field(default_factory=lambda: [
        "Temperature", "Heating_Days", "Cooling_Days", "Precipitation"
    ])

    # Other potential predictors
    other_columns: List[str] = field(default_factory=lambda: [
        "Carbon_Price", "Oil_Price", "Gas_Price", "Electricity_Price"
    ])

    # Column name aliases (for flexible mapping)
    aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "date": ["Quarter", "Date", "Period", "Time", "Qtr"],
        "target": ["CO2e", "CO2", "Emissions", "CO2_emissions", "GHG", "Carbon"],
        "gdp": ["GDP", "gdp", "Gross_Domestic_Product"],
        "population": ["Population", "Pop", "population"],
    })

    def get_all_feature_columns(self) -> List[str]:
        """Get all feature column names."""
        return (
            self.economic_columns +
            self.energy_columns +
            self.climate_columns +
            self.other_columns
        )

    def find_column(self, df_columns: List[str], logical_name: str) -> Optional[str]:
        """
        Find actual column name in dataframe for a logical name.

        Args:
            df_columns: List of column names in the dataframe
            logical_name: Logical name to search for

        Returns:
            Actual column name if found, None otherwise
        """
        # Direct match (case-insensitive)
        for col in df_columns:
            if col.lower() == logical_name.lower():
                return col

        # Check aliases
        if logical_name.lower() in self.aliases:
            for alias in self.aliases[logical_name.lower()]:
                for col in df_columns:
                    if col.lower() == alias.lower():
                        return col

        # Partial match
        for col in df_columns:
            if logical_name.lower() in col.lower():
                return col

        return None

    def auto_detect_columns(self, df_columns: List[str]) -> Dict[str, str]:
        """
        Auto-detect column mappings from dataframe columns.

        Args:
            df_columns: List of column names in the dataframe

        Returns:
            Dictionary mapping logical names to actual column names
        """
        mappings = {}

        # Find date column
        for alias in self.aliases.get('date', []):
            for col in df_columns:
                if alias.lower() in col.lower():
                    mappings['date'] = col
                    break
            if 'date' in mappings:
                break

        # Find target column
        for alias in self.aliases.get('target', []):
            for col in df_columns:
                if alias.lower() in col.lower():
                    mappings['target'] = col
                    break
            if 'target' in mappings:
                break

        # Find feature columns
        mappings['features'] = []
        for col in df_columns:
            if col in [mappings.get('date'), mappings.get('target')]:
                continue
            mappings['features'].append(col)

        return mappings

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            'date_column': self.date_column,
            'target_column': self.target_column,
            'economic_columns': self.economic_columns,
            'energy_columns': self.energy_columns,
            'climate_columns': self.climate_columns,
            'other_columns': self.other_columns,
            'aliases': self.aliases
        }

    def save(self, path: str):
        """Save schema to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'DataSchema':
        """Load schema from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


def create_default_schema() -> DataSchema:
    """Create default schema for CO2 data."""
    return DataSchema()


def validate_schema(df, schema: DataSchema) -> Dict[str, Any]:
    """
    Validate a dataframe against a schema.

    Args:
        df: pandas DataFrame
        schema: DataSchema instance

    Returns:
        Validation results dictionary
    """
    import pandas as pd

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'mappings': {},
        'missing_columns': [],
        'extra_columns': []
    }

    df_columns = list(df.columns)

    # Check date column
    date_col = schema.find_column(df_columns, 'date')
    if date_col:
        results['mappings']['date'] = date_col
    else:
        results['errors'].append(f"Date column not found. Expected one of: {schema.aliases.get('date', [])}")
        results['valid'] = False

    # Check target column
    target_col = schema.find_column(df_columns, 'target')
    if target_col:
        results['mappings']['target'] = target_col
    else:
        results['errors'].append(f"Target column not found. Expected one of: {schema.aliases.get('target', [])}")
        results['valid'] = False

    # Check feature columns
    expected_features = schema.get_all_feature_columns()
    found_features = []
    missing_features = []

    for feat in expected_features:
        actual = schema.find_column(df_columns, feat)
        if actual:
            found_features.append(actual)
            results['mappings'][feat] = actual
        else:
            missing_features.append(feat)

    if missing_features:
        results['warnings'].append(f"Some expected features not found: {missing_features[:5]}...")

    # Find extra columns
    mapped_cols = list(results['mappings'].values())
    extra = [c for c in df_columns if c not in mapped_cols]
    if extra:
        results['extra_columns'] = extra
        results['warnings'].append(f"Additional columns found: {extra[:5]}...")

    results['missing_columns'] = missing_features

    return results
