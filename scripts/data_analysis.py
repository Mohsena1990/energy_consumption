#!/usr/bin/env python3
"""
Comprehensive Data Analysis Script for CO2 Forecasting
======================================================
Analyzes data distributions, outliers, stationarity, and correlations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, kurtosis, skew
from scipy.stats import iqr as scipy_iqr

# Output directory
OUTPUT_DIR = Path("outputs/data_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load raw and processed data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df_raw = None
    df_clean = None

    # Try to load raw Excel data
    raw_path = Path("data 1999-2025Q1.xlsx")
    try:
        df_raw = pd.read_excel(raw_path, header=2)
        print(f"Raw data shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
    except Exception as e:
        print(f"Could not load Excel file: {e}")

    # Load processed data if available
    processed_path = Path("data/processed/df_clean.parquet")
    if processed_path.exists():
        df_clean = pd.read_parquet(processed_path)
        print(f"Processed data shape: {df_clean.shape}")
        print(f"Processed columns: {list(df_clean.columns)}")

    # Use processed data if raw is not available
    if df_raw is None and df_clean is not None:
        df_raw = df_clean.copy()
        print("Using processed data for analysis")

    return df_raw, df_clean


def basic_statistics(df, name="Data"):
    """Generate basic statistics for the dataset."""
    print("\n" + "=" * 70)
    print(f"BASIC STATISTICS: {name}")
    print("=" * 70)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats_df = pd.DataFrame()

    for col in numeric_cols:
        data = df[col].dropna()

        stats_df[col] = {
            'count': len(data),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            '25%': data.quantile(0.25),
            '50%': data.quantile(0.50),
            '75%': data.quantile(0.75),
            'max': data.max(),
            'range': data.max() - data.min(),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'cv': (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan,  # Coefficient of variation
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
        }

    stats_df = stats_df.T
    print(stats_df.to_string())
    stats_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_basic_stats.csv")

    return stats_df


def distribution_analysis(df, name="Data"):
    """Analyze distribution characteristics of each variable."""
    print("\n" + "=" * 70)
    print(f"DISTRIBUTION ANALYSIS: {name}")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in numeric_cols:
        data = df[col].dropna()

        if len(data) < 8:
            print(f"  {col}: Insufficient data for distribution tests")
            continue

        result = {'variable': col}

        # Skewness interpretation
        skewness = skew(data)
        result['skewness'] = skewness
        if abs(skewness) < 0.5:
            result['skew_interpretation'] = 'Approximately symmetric'
        elif skewness > 0:
            result['skew_interpretation'] = 'Right-skewed (positive)'
        else:
            result['skew_interpretation'] = 'Left-skewed (negative)'

        # Kurtosis interpretation
        kurt = kurtosis(data)
        result['kurtosis'] = kurt
        if abs(kurt) < 0.5:
            result['kurt_interpretation'] = 'Mesokurtic (normal-like)'
        elif kurt > 0:
            result['kurt_interpretation'] = 'Leptokurtic (heavy tails)'
        else:
            result['kurt_interpretation'] = 'Platykurtic (light tails)'

        # Normality tests
        try:
            # Shapiro-Wilk test (best for small samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = shapiro(data)
                result['shapiro_stat'] = shapiro_stat
                result['shapiro_p'] = shapiro_p
                result['shapiro_normal'] = 'Yes' if shapiro_p > 0.05 else 'No'
        except Exception as e:
            result['shapiro_normal'] = 'Error'

        try:
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(data)
            result['jarque_bera_stat'] = jb_stat
            result['jarque_bera_p'] = jb_p
            result['jb_normal'] = 'Yes' if jb_p > 0.05 else 'No'
        except Exception as e:
            result['jb_normal'] = 'Error'

        # Distribution type suggestion
        if result.get('shapiro_normal') == 'Yes' or result.get('jb_normal') == 'Yes':
            result['distribution_type'] = 'Normal/Gaussian'
        elif skewness > 1:
            result['distribution_type'] = 'Log-normal or Exponential'
        elif skewness < -1:
            result['distribution_type'] = 'Left-skewed (consider reflection)'
        else:
            result['distribution_type'] = 'Non-normal (consider transformation)'

        results.append(result)

        print(f"\n  {col}:")
        print(f"    Skewness: {skewness:.4f} ({result['skew_interpretation']})")
        print(f"    Kurtosis: {kurt:.4f} ({result['kurt_interpretation']})")
        print(f"    Shapiro-Wilk p-value: {result.get('shapiro_p', 'N/A'):.4f if isinstance(result.get('shapiro_p'), float) else 'N/A'}")
        print(f"    Normal: {result.get('shapiro_normal', 'N/A')}")
        print(f"    Suggested type: {result['distribution_type']}")

    dist_df = pd.DataFrame(results)
    dist_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_distribution.csv", index=False)

    return dist_df


def outlier_analysis(df, name="Data"):
    """Detect outliers using multiple methods."""
    print("\n" + "=" * 70)
    print(f"OUTLIER ANALYSIS: {name}")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    outlier_details = {}

    for col in numeric_cols:
        data = df[col].dropna()

        if len(data) < 4:
            continue

        result = {'variable': col, 'n_observations': len(data)}

        # Method 1: IQR method (1.5 * IQR)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        outliers_iqr = data[(data < lower_bound_iqr) | (data > upper_bound_iqr)]
        result['iqr_outliers'] = len(outliers_iqr)
        result['iqr_outlier_pct'] = (len(outliers_iqr) / len(data)) * 100
        result['iqr_lower'] = lower_bound_iqr
        result['iqr_upper'] = upper_bound_iqr

        # Method 2: Z-score method (|z| > 3)
        z_scores = np.abs(stats.zscore(data))
        outliers_zscore = data[z_scores > 3]
        result['zscore_outliers'] = len(outliers_zscore)
        result['zscore_outlier_pct'] = (len(outliers_zscore) / len(data)) * 100

        # Method 3: Modified Z-score (using MAD)
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad != 0:
            modified_z = 0.6745 * (data - median) / mad
            outliers_mad = data[np.abs(modified_z) > 3.5]
            result['mad_outliers'] = len(outliers_mad)
            result['mad_outlier_pct'] = (len(outliers_mad) / len(data)) * 100
        else:
            result['mad_outliers'] = 0
            result['mad_outlier_pct'] = 0

        # Store outlier indices for detailed view
        outlier_indices = outliers_iqr.index.tolist()
        if outlier_indices:
            outlier_details[col] = {
                'indices': outlier_indices,
                'values': outliers_iqr.tolist(),
                'bounds': (lower_bound_iqr, upper_bound_iqr)
            }

        results.append(result)

        print(f"\n  {col}:")
        print(f"    IQR method: {result['iqr_outliers']} outliers ({result['iqr_outlier_pct']:.1f}%)")
        print(f"      Bounds: [{lower_bound_iqr:.4f}, {upper_bound_iqr:.4f}]")
        print(f"    Z-score method: {result['zscore_outliers']} outliers ({result['zscore_outlier_pct']:.1f}%)")
        print(f"    MAD method: {result['mad_outliers']} outliers ({result['mad_outlier_pct']:.1f}%)")
        if outlier_indices:
            print(f"    Outlier values: {outliers_iqr.tolist()[:5]}{'...' if len(outliers_iqr) > 5 else ''}")

    outlier_df = pd.DataFrame(results)
    outlier_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_outliers.csv", index=False)

    # Save detailed outlier info
    if outlier_details:
        detail_rows = []
        for col, info in outlier_details.items():
            for idx, val in zip(info['indices'], info['values']):
                detail_rows.append({
                    'variable': col,
                    'index': idx,
                    'value': val,
                    'lower_bound': info['bounds'][0],
                    'upper_bound': info['bounds'][1]
                })
        if detail_rows:
            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_outlier_details.csv", index=False)

    return outlier_df, outlier_details


def stationarity_analysis(df, name="Data"):
    """Analyze time series stationarity."""
    print("\n" + "=" * 70)
    print(f"STATIONARITY ANALYSIS: {name}")
    print("=" * 70)

    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        print("  statsmodels not available, skipping stationarity tests")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in numeric_cols:
        data = df[col].dropna()

        if len(data) < 12:  # Need enough data points
            continue

        result = {'variable': col}

        try:
            # ADF test (null: non-stationary)
            adf_result = adfuller(data, autolag='AIC')
            result['adf_statistic'] = adf_result[0]
            result['adf_pvalue'] = adf_result[1]
            result['adf_stationary'] = 'Yes' if adf_result[1] < 0.05 else 'No'
        except Exception as e:
            result['adf_stationary'] = 'Error'

        try:
            # KPSS test (null: stationary)
            kpss_result = kpss(data, regression='c', nlags='auto')
            result['kpss_statistic'] = kpss_result[0]
            result['kpss_pvalue'] = kpss_result[1]
            result['kpss_stationary'] = 'Yes' if kpss_result[1] > 0.05 else 'No'
        except Exception as e:
            result['kpss_stationary'] = 'Error'

        # Trend detection (simple linear regression)
        try:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            result['trend_slope'] = slope
            result['trend_r2'] = r_value ** 2
            result['trend_pvalue'] = p_value
            result['has_trend'] = 'Yes' if p_value < 0.05 else 'No'
        except Exception:
            result['has_trend'] = 'Error'

        results.append(result)

        print(f"\n  {col}:")
        print(f"    ADF test: p={result.get('adf_pvalue', 'N/A'):.4f if isinstance(result.get('adf_pvalue'), float) else 'N/A'}, Stationary: {result.get('adf_stationary', 'N/A')}")
        print(f"    KPSS test: p={result.get('kpss_pvalue', 'N/A'):.4f if isinstance(result.get('kpss_pvalue'), float) else 'N/A'}, Stationary: {result.get('kpss_stationary', 'N/A')}")
        print(f"    Trend: {result.get('has_trend', 'N/A')} (slope={result.get('trend_slope', 0):.6f}, R²={result.get('trend_r2', 0):.4f})")

    station_df = pd.DataFrame(results)
    station_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_stationarity.csv", index=False)

    return station_df


def correlation_analysis(df, name="Data"):
    """Analyze correlations and multicollinearity."""
    print("\n" + "=" * 70)
    print(f"CORRELATION ANALYSIS: {name}")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print("  Not enough numeric columns for correlation analysis")
        return None, None

    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3).to_string())
    corr_matrix.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_correlation.csv")

    # High correlations (excluding diagonal)
    high_corr = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append({
                    'var1': numeric_cols[i],
                    'var2': numeric_cols[j],
                    'correlation': corr_val,
                    'strength': 'Very High' if abs(corr_val) > 0.9 else 'High'
                })

    if high_corr:
        print("\n  High Correlations (|r| > 0.7):")
        for hc in high_corr:
            print(f"    {hc['var1']} <-> {hc['var2']}: {hc['correlation']:.4f} ({hc['strength']})")
        high_corr_df = pd.DataFrame(high_corr)
        high_corr_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_high_correlations.csv", index=False)

    # VIF calculation
    print("\n  Variance Inflation Factors (VIF):")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Remove columns with zero variance
        valid_cols = [c for c in numeric_cols if df[c].std() > 0]
        X = df[valid_cols].dropna()

        if len(X) > len(valid_cols):
            vif_data = []
            for i, col in enumerate(valid_cols):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_data.append({'variable': col, 'VIF': vif})
                    interpretation = 'OK' if vif < 5 else ('Moderate' if vif < 10 else 'High multicollinearity')
                    print(f"    {col}: {vif:.2f} ({interpretation})")
                except Exception:
                    pass

            if vif_data:
                vif_df = pd.DataFrame(vif_data)
                vif_df.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_vif.csv", index=False)
    except ImportError:
        print("    statsmodels not available for VIF calculation")

    return corr_matrix, high_corr


def seasonal_analysis(df, date_col=None, target_col='CO2e', name="Data"):
    """Analyze seasonal patterns."""
    print("\n" + "=" * 70)
    print(f"SEASONAL ANALYSIS: {name}")
    print("=" * 70)

    if target_col not in df.columns:
        print(f"  Target column '{target_col}' not found")
        return None

    # Try to identify quarter from data
    if 'Time_Period' in df.columns:
        try:
            # Parse quarter from Time_Period (e.g., 1999.1 -> Q1 1999)
            quarters = df['Time_Period'].apply(lambda x: int((x % 1) * 10) if pd.notna(x) else np.nan)
            quarters = quarters.replace({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
        except:
            quarters = None
    else:
        quarters = None

    if quarters is not None and target_col in df.columns:
        seasonal_stats = df.groupby(quarters)[target_col].agg(['mean', 'std', 'min', 'max', 'count'])
        print("\n  Seasonal Statistics by Quarter:")
        print(seasonal_stats.to_string())
        seasonal_stats.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_seasonal.csv")

        # Check for seasonal differences
        try:
            from scipy.stats import f_oneway
            q_groups = [df[quarters == q][target_col].dropna() for q in ['Q1', 'Q2', 'Q3', 'Q4'] if q in quarters.values]
            if len(q_groups) >= 2 and all(len(g) > 0 for g in q_groups):
                f_stat, p_value = f_oneway(*q_groups)
                print(f"\n  ANOVA test for seasonal differences:")
                print(f"    F-statistic: {f_stat:.4f}")
                print(f"    p-value: {p_value:.4f}")
                print(f"    Significant seasonal effect: {'Yes' if p_value < 0.05 else 'No'}")
        except Exception as e:
            print(f"  Could not perform ANOVA: {e}")

    return None


def generate_summary_report(df, stats_df, dist_df, outlier_df, station_df, corr_matrix):
    """Generate a summary report with recommendations."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT AND RECOMMENDATIONS")
    print("=" * 70)

    report = []

    report.append("DATA ANALYSIS SUMMARY")
    report.append("=" * 50)
    report.append(f"\nDataset: {df.shape[0]} observations, {df.shape[1]} variables\n")

    # Distribution summary
    report.append("\n1. DISTRIBUTION ANALYSIS")
    report.append("-" * 30)
    if dist_df is not None:
        non_normal = dist_df[dist_df['shapiro_normal'] == 'No']['variable'].tolist() if 'shapiro_normal' in dist_df.columns else []
        if non_normal:
            report.append(f"  Non-normal variables: {', '.join(non_normal)}")
            report.append("  RECOMMENDATION: Consider log/Box-Cox transformation for right-skewed variables")

        highly_skewed = dist_df[abs(dist_df['skewness']) > 1]['variable'].tolist() if 'skewness' in dist_df.columns else []
        if highly_skewed:
            report.append(f"  Highly skewed variables: {', '.join(highly_skewed)}")

    # Outlier summary
    report.append("\n2. OUTLIER ANALYSIS")
    report.append("-" * 30)
    if outlier_df is not None:
        high_outlier_vars = outlier_df[outlier_df['iqr_outlier_pct'] > 5]['variable'].tolist() if 'iqr_outlier_pct' in outlier_df.columns else []
        if high_outlier_vars:
            report.append(f"  Variables with >5% outliers: {', '.join(high_outlier_vars)}")
            report.append("  RECOMMENDATION: Investigate outliers - may be data errors or genuine extreme values")
        else:
            report.append("  All variables have acceptable outlier percentages (<5%)")

    # Stationarity summary
    report.append("\n3. STATIONARITY ANALYSIS")
    report.append("-" * 30)
    if station_df is not None:
        non_stationary = station_df[station_df['adf_stationary'] == 'No']['variable'].tolist() if 'adf_stationary' in station_df.columns else []
        if non_stationary:
            report.append(f"  Non-stationary variables: {', '.join(non_stationary)}")
            report.append("  RECOMMENDATION: Consider differencing or detrending for time series models")

        trending = station_df[station_df['has_trend'] == 'Yes']['variable'].tolist() if 'has_trend' in station_df.columns else []
        if trending:
            report.append(f"  Variables with significant trend: {', '.join(trending)}")

    # Correlation summary
    report.append("\n4. CORRELATION/MULTICOLLINEARITY")
    report.append("-" * 30)
    if corr_matrix is not None:
        # Find high correlations
        high_corr_pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if abs(corr_matrix.iloc[i,j]) > 0.8:
                    high_corr_pairs.append(f"{cols[i]}-{cols[j]}")
        if high_corr_pairs:
            report.append(f"  Highly correlated pairs (|r|>0.8): {', '.join(high_corr_pairs[:5])}")
            report.append("  RECOMMENDATION: Consider removing one variable from each pair or use regularization")

    # General recommendations
    report.append("\n5. PREPROCESSING RECOMMENDATIONS")
    report.append("-" * 30)
    report.append("  a) Missing values: Check and impute using appropriate methods")
    report.append("  b) Scaling: StandardScaler for normal distributions, RobustScaler for outliers")
    report.append("  c) Feature engineering: Create lag features, rolling statistics")
    report.append("  d) Target transformation: Log transform if target is right-skewed")

    # Print and save report
    report_text = '\n'.join(report)
    print(report_text)

    with open(OUTPUT_DIR / "analysis_summary_report.txt", 'w') as f:
        f.write(report_text)

    return report_text


def main():
    """Run complete data analysis."""
    print("=" * 70)
    print("COMPREHENSIVE DATA ANALYSIS FOR CO2 FORECASTING")
    print("=" * 70)

    # Load data
    df_raw, df_clean = load_data()

    # Use raw data for analysis
    df = df_raw.copy()

    # Basic statistics
    stats_df = basic_statistics(df, "Raw Data")

    # Distribution analysis
    dist_df = distribution_analysis(df, "Raw Data")

    # Outlier analysis
    outlier_df, outlier_details = outlier_analysis(df, "Raw Data")

    # Stationarity analysis
    station_df = stationarity_analysis(df, "Raw Data")

    # Correlation analysis
    corr_matrix, high_corr = correlation_analysis(df, "Raw Data")

    # Seasonal analysis
    seasonal_analysis(df, target_col='CO2e', name="Raw Data")

    # Generate summary report
    generate_summary_report(df, stats_df, dist_df, outlier_df, station_df, corr_matrix)

    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
