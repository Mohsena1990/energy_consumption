"""
Data loading utilities for CO2 forecasting framework.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from ..core.config import Config, DataConfig
from ..core.utils import quarter_to_date, ensure_quarterly_index
from ..core.logging_utils import get_logger
from .schema import DataSchema, validate_schema, create_default_schema


def load_excel_data(
    path: str,
    sheet_name: Optional[str] = None,
    date_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from Excel file, robust to cover sheets and offset headers.
    """
    logger = get_logger()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from: {path}")

    xl = pd.ExcelFile(path)

    # If user didn't specify a sheet, prefer a typical data sheet
    chosen_sheet = sheet_name
    if chosen_sheet is None:
        if "Sheet1" in xl.sheet_names:
            chosen_sheet = "Sheet1"
        else:
            # fallback: last sheet often holds the table
            chosen_sheet = xl.sheet_names[-1]

    # --- detect header row by scanning first ~30 rows for table headers ---
    preview = pd.read_excel(path, sheet_name=chosen_sheet, header=None, nrows=30)

    header_row = 0
    target_markers = {"CO2e", "CO2", "Emissions", "Carbon"}
    date_markers = {"Time_Period", "Quarter", "Date", "Period", "Time", "Qtr"}

    for r in range(len(preview)):
        row_vals = set(preview.iloc[r].astype(str).str.strip().tolist())
        if (row_vals & target_markers) and (row_vals & date_markers):
            header_row = r
            break

    df = pd.read_excel(path, sheet_name=chosen_sheet, header=header_row)

    logger.info(f"Loaded sheet: {chosen_sheet} (header row: {header_row})")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df



def prepare_data(
    df: pd.DataFrame,
    schema: DataSchema,
    config: DataConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Prepare data according to schema and config.

    Args:
        df: Raw DataFrame
        schema: Data schema
        config: Data configuration

    Returns:
        Prepared DataFrame and column mappings
    """
    logger = get_logger()
    df = df.copy()
    if "Time_Period" in df.columns:
        df = df.rename(columns={"Time_Period": "Quarter"})
    # Validate schema
    validation = validate_schema(df, schema)
    if not validation['valid']:
        raise ValueError(f"Schema validation failed: {validation['errors']}")

    if validation['warnings']:
        for w in validation['warnings']:
            logger.warning(w)

    mappings = validation['mappings']
    df = df.copy()

    # Handle date column
    date_col = mappings.get('date')
    if date_col and date_col in df.columns:
        # Convert to datetime
        try:

            def parse_year_quarter(x):
                # Handles 1999.1, 1999.2, ..., 2025.1
                year = int(np.floor(x))
                quarter = int(round((x - year) * 10))
                month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
                return pd.Timestamp(year=year, month=month, day=1)

            df['date'] = df[date_col].apply(parse_year_quarter)



            # df['date'] = df[date_col].apply(quarter_to_date)
        except Exception as e:
            logger.warning(f"Could not parse date column: {e}")
            # Try pandas parsing
            df['date'] = pd.to_datetime(df[date_col])

        df = df.set_index('date')
        df = df.drop(columns=[date_col], errors='ignore')

    # Sort by date
    df = df.sort_index()

    # Handle target column
    target_col = mappings.get('target')
    if target_col and target_col in df.columns and target_col != config.target_column:
        df = df.rename(columns={target_col: config.target_column})
        mappings['target'] = config.target_column

    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Target column: {config.target_column}")

    return df, mappings


def create_quarterly_index(start: str, end: str) -> pd.DatetimeIndex:
    """
    Create quarterly datetime index.

    Args:
        start: Start quarter (e.g., "1999Q1")
        end: End quarter (e.g., "2025Q1")

    Returns:
        DatetimeIndex with quarterly frequency
    """
    start_date = quarter_to_date(start)
    end_date = quarter_to_date(end)

    return pd.date_range(start=start_date, end=end_date, freq='QS')


def align_to_quarterly(
    df: pd.DataFrame,
    start: str = "1999Q1",
    end: str = "2025Q1"
) -> pd.DataFrame:
    """
    Align DataFrame to quarterly frequency.

    Args:
        df: Input DataFrame with datetime index
        start: Start quarter
        end: End quarter

    Returns:
        DataFrame aligned to quarterly frequency
    """
    logger = get_logger()

    # Create target index
    target_index = create_quarterly_index(start, end)

    # Reindex to quarterly
    df_quarterly = df.reindex(target_index)

    # Log missing values
    missing = df_quarterly.isnull().any(axis=1).sum()
    if missing > 0:
        logger.warning(f"Missing data for {missing} quarters after alignment")

    return df_quarterly


def load_and_prepare_data(
    config: Config,
    schema: Optional[DataSchema] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to load and prepare data.

    Args:
        config: Configuration object
        schema: Data schema (optional, will use default if None)

    Returns:
        Tuple of (prepared DataFrame, metadata dict)
    """
    logger = get_logger()

    if schema is None:
        schema = create_default_schema()

    # Load raw data
    df_raw = load_excel_data(
        config.data.input_path,
        sheet_name=config.data.sheet_name
    )

    # Prepare data
    df, mappings = prepare_data(df_raw, schema, config.data)

    # Create metadata
    metadata = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'date_range': {
            'start': str(df.index.min()),
            'end': str(df.index.max())
        },
        'columns': list(df.columns),
        'mappings': mappings,
        'target_column': config.data.target_column
    }

    logger.info(f"Data prepared: {metadata['n_rows']} rows, {metadata['n_columns']} columns")

    return df, metadata


def _has_parquet_support() -> bool:
    """Check if parquet support is available."""
    try:
        import pyarrow
        return True
    except ImportError:
        pass
    try:
        import fastparquet
        return True
    except ImportError:
        pass
    return False


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Path,
    name: str = "df_processed"
) -> Path:
    """
    Save processed DataFrame to parquet (or CSV as fallback).

    Args:
        df: DataFrame to save
        output_dir: Output directory
        name: File name (without extension)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_parquet_support():
        output_path = output_dir / f"{name}.parquet"
        df.to_parquet(output_path)
    else:
        # Fallback to CSV
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path)

    return output_path


def load_processed_data(path: str) -> pd.DataFrame:
    """Load processed DataFrame from parquet or CSV."""
    path = Path(path)

    # Try parquet first
    if path.suffix == '.parquet':
        if _has_parquet_support():
            return pd.read_parquet(path)
        else:
            # Try CSV fallback
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return df
            raise ImportError(
                f"Cannot read {path}: parquet support not available. "
                "Install pyarrow: pip install pyarrow"
            )
    elif path.suffix == '.csv':
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    else:
        # Try both extensions
        parquet_path = Path(str(path) + '.parquet')
        csv_path = Path(str(path) + '.csv')

        if parquet_path.exists() and _has_parquet_support():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        elif parquet_path.exists():
            raise ImportError(
                f"Cannot read {parquet_path}: parquet support not available. "
                "Install pyarrow: pip install pyarrow"
            )
        else:
            raise FileNotFoundError(f"No data file found at {path}")
