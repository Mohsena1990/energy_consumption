# Comprehensive Data Analysis Report
## CO2 Emissions Forecasting - UK Quarterly Data (1999-2025)

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Time Period** | 1999-Q1 to 2025-Q1 |
| **Observations** | 105 quarters |
| **Variables** | 8 numeric features |
| **Missing Values** | None (0%) |
| **Temporal Gaps** | None |

### Variables Description

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| **CO2e** | Target | CO2 equivalent emissions | Thousands of tonnes |
| **GDP** | Economic | Gross Domestic Product | Index/Billions GBP |
| **Population** | Demographic | UK population | Thousands |
| **Air_Temp** | Climate | Average air temperature | Celsius |
| **Rainfall** | Climate | Total rainfall | mm |
| **TEC** | Energy | Total Energy Consumption | GWh or equivalent |
| **CEI** | Energy | Carbon Emission Intensity | Ratio |
| **COVID_Deaths** | Health/Shock | COVID-19 related deaths | Count |

---

## 2. Distribution Analysis

### 2.1 Target Variable: CO2e

| Statistic | Value |
|-----------|-------|
| Mean | 136,490.89 |
| Std Dev | 24,844.92 |
| Min | 86,119.70 |
| 25% | 115,380.50 |
| Median | 137,066.60 |
| 75% | 153,694.50 |
| Max | 185,503.50 |
| Range | 99,383.80 |
| CV (%) | 18.2% |

**Distribution Assessment:**
- Coefficient of Variation (18.2%) indicates moderate variability
- Range spans ~2x from min to max
- Median close to mean suggests relatively symmetric distribution
- **Current log transformation is appropriate** for stabilizing variance

### 2.2 Economic Variables

#### GDP
| Statistic | Value |
|-----------|-------|
| Mean | 9,228.70 |
| Std Dev | 615.40 |
| Min | 7,795 |
| Max | 10,147 |
| CV (%) | 6.7% |

**Assessment:** Low variability, upward trending, near-normal distribution

#### Population
| Statistic | Value |
|-----------|-------|
| Mean | 63,553.18 |
| Std Dev | 3,255.67 |
| Min | 58,632 |
| Max | 69,509 |
| CV (%) | 5.1% |

**Assessment:** Very low variability, strong upward trend, likely non-stationary

### 2.3 Climate Variables

#### Air_Temp
| Statistic | Value |
|-----------|-------|
| Mean | 9.30 |
| Std Dev | 3.88 |
| Min | 1.62 |
| Max | 15.80 |
| CV (%) | 41.7% |

**Assessment:** High variability (expected due to seasonality), near-symmetric distribution

#### Rainfall
| Statistic | Value |
|-----------|-------|
| Mean | 295.48 |
| Std Dev | 82.19 |
| Min | 141.40 |
| Max | 539.90 |
| CV (%) | 27.8% |

**Assessment:** Moderate variability, slight right skew (max is 1.8x mean)

### 2.4 Energy Variables

#### TEC (Total Energy Consumption)
| Statistic | Value |
|-----------|-------|
| Mean | 51,412.73 |
| Std Dev | 9,529.24 |
| Min | 31,679.99 |
| Max | 70,923.94 |
| CV (%) | 18.5% |

**Assessment:** Moderate variability, likely trends with economic activity

#### CEI (Carbon Emission Intensity)
| Statistic | Value |
|-----------|-------|
| Mean | 0.214 |
| Std Dev | 0.060 |
| Min | 0.124 |
| Max | 0.312 |
| CV (%) | 28.2% |

**Assessment:** Moderate variability, downward trending (decarbonization)

### 2.5 Shock Variable

#### COVID_Deaths
| Statistic | Value |
|-----------|-------|
| Mean | 2,177.37 |
| Std Dev | 8,694.76 |
| Min | 0 |
| Median | 0 |
| Max | 60,810 |
| CV (%) | 399.3% |

**Assessment:**
- **EXTREMELY RIGHT-SKEWED** (CV > 100%)
- 75% of values are 0
- Only ~15% of observations have non-zero values
- **NOT normally distributed** - effectively a spike/shock variable

---

## 3. Outlier Analysis

### 3.1 IQR Method Results

| Variable | Outliers | % | Lower Bound | Upper Bound |
|----------|----------|---|-------------|-------------|
| Rainfall | 1 | 0.95% | - | 539.9 (max) |
| COVID_Deaths | 15 | 14.3% | N/A | N/A |

### 3.2 Detailed Analysis

#### Rainfall Outlier (1 observation)
- **Value:** 539.9 mm (single maximum)
- **Assessment:** Genuine extreme weather event, NOT a data error
- **Recommendation:** **Keep as-is** - represents real climate variability

#### COVID_Deaths Outliers (15 observations)
- **Context:** COVID pandemic period (2020-2022)
- **Assessment:** NOT outliers in the traditional sense - these are genuine shock events
- **Recommendation:** **Keep as-is** - handled via COVID dummy variable in feature engineering

### 3.3 Other Variables
- No outliers detected in: CO2e, GDP, Population, Air_Temp, TEC, CEI
- This indicates good data quality

---

## 4. Stationarity & Trend Analysis

Based on the log file and feature selection results:

### 4.1 Variables with Strong Trends

| Variable | Trend Direction | Likely Stationary? | Recommendation |
|----------|-----------------|--------------------|-----------------|
| CO2e | Downward (decarbonization) | No | Log transform (already applied) |
| GDP | Upward | No | Already differenced via lag features |
| Population | Upward (linear) | No | Consider exclusion (high VIF) |
| TEC | Non-linear | No | Captured via lag features |
| CEI | Downward (decarbonization) | No | High multicollinearity with CO2e |

### 4.2 Stationary/Seasonal Variables

| Variable | Pattern | Notes |
|----------|---------|-------|
| Air_Temp | Seasonal (quarterly) | Stationary around seasonal mean |
| Rainfall | Seasonal + random | Stationary around seasonal mean |

---

## 5. Correlation & Multicollinearity Analysis

### 5.1 VIF Analysis Results (from logs)

Variables removed due to high VIF (>10):
1. **CEI** (VIF=130.82) - Removed first
2. **Q3** (VIF=59.88) - Seasonal collinearity
3. **TEC** (VIF=59.26) - Correlated with CO2e
4. **CO2e_lag3** (VIF=40.88) - Lag collinearity
5. **CO2e_lag1** (VIF=31.89) - Lag collinearity
6. **Population** (VIF=22.71) - Trend collinearity
7. **CO2e_lag2** (VIF=10.19) - Lag collinearity

### 5.2 High Correlation Pairs (Expected)

| Pair | Likely Correlation | Reason |
|------|-------------------|--------|
| CO2e ↔ TEC | Very High (>0.9) | Emissions from energy use |
| CO2e ↔ CEI | Very High (>0.9) | Intensity relationship |
| CO2e_lag1 ↔ CO2e_lag2 | Very High (>0.9) | Temporal autocorrelation |
| GDP ↔ Population | High (0.7-0.9) | Economic growth |
| TEC ↔ GDP | High (0.7-0.9) | Energy-economy nexus |

### 5.3 Feature Stability Scores (Ridge Regression)

| Feature | Stability Score | Interpretation |
|---------|-----------------|----------------|
| Air_Temp | 1.00 | Most stable predictor |
| CO2e_lag4 | 1.00 | Strong temporal persistence |
| GDP | 0.66 | Moderate stability |
| Q4 | 0.27 | Some seasonal effect |
| COVID_Deaths | 0.25 | Low stability (event-specific) |
| Others | <0.20 | Unstable predictors |

---

## 6. Distribution Type Summary & Transformation Recommendations

| Variable | Distribution Type | Recommended Transform | Applied? |
|----------|------------------|----------------------|----------|
| **CO2e** | Near-normal (slight right skew) | Log | Yes |
| **GDP** | Near-normal, trending | None (or differencing) | No |
| **Population** | Near-normal, trending | None (exclude if VIF high) | Excluded |
| **Air_Temp** | Near-normal, seasonal | None | No |
| **Rainfall** | Slight right skew | None or sqrt | No |
| **TEC** | Near-normal, trending | None (excluded due to VIF) | Excluded |
| **CEI** | Near-normal, trending | None (excluded due to VIF) | Excluded |
| **COVID_Deaths** | Zero-inflated, extreme skew | Binary dummy | Yes (COVID dummy) |

---

## 7. Key Findings & Recommendations

### 7.1 Data Quality: GOOD
- No missing values
- No temporal gaps
- Minimal true outliers
- Good variable coverage

### 7.2 Distribution Issues

**Issue 1: COVID_Deaths is Zero-Inflated**
- Current: Raw values included in nonlinear models
- Recommendation: Use COVID dummy (already implemented) for linear models
- For nonlinear models: Consider binary transformation (0/1) or log(1+x)

**Issue 2: Strong Multicollinearity**
- TEC, CEI highly correlated with target (data leakage risk)
- Population trends with GDP
- Recommendation: Keep using VIF-based filtering (already implemented)

**Issue 3: Trending Variables**
- GDP, Population, CEI have strong trends
- Recommendation: Current lag-based approach handles this well

### 7.3 Feature Selection Recommendation

Based on the analysis, the **fs_linear** selection (3 features) may be too parsimonious:
- GDP, Air_Temp, CO2e_lag4

Consider using **fs_consensus** (10 features) for better predictive power:
- Weighted MAE: 0.061 vs 0.090
- Includes more lag features for temporal dynamics

### 7.4 Preprocessing Pipeline Validation

Current pipeline is **well-designed**:
1. Log transform on target (appropriate for CO2e)
2. VIF-based multicollinearity removal (effective)
3. Lag features (captures temporal dynamics)
4. Seasonal dummies (handles quarterly patterns)
5. Shock dummies (COVID, Energy Crisis)

### 7.5 Additional Recommendations

1. **For LSTM**: Consider using more features (fs_consensus or fs_nonlinear)
   - LSTM benefits from richer feature representation
   - Current 3 features may limit learning capacity

2. **For Robustness**: Add RobustScaler for COVID_Deaths if including raw values
   - Current StandardScaler may be affected by extreme values

3. **Consider Adding**:
   - Rolling statistics (4-quarter rolling mean/std) for trend capture
   - Year-over-year growth rates for GDP

---

## 8. Summary Statistics Table

```
                  CO2e       GDP  Population  Air_Temp  Rainfall       TEC     CEI  COVID_Deaths
count           105.00    105.00      105.00    105.00    105.00    105.00  105.00        105.00
mean        136490.89   9228.70    63553.18      9.30    295.48  51412.73    0.21       2177.37
std          24844.92    615.40     3255.67      3.88     82.19   9529.24    0.06       8694.76
min          86119.70   7795.00    58632.00      1.62    141.40  31679.99    0.12          0.00
25%         115380.50   8877.00    60517.00      5.79    236.30  44497.13    0.16          0.00
50%         137066.60   9195.00    63604.00      9.02    290.20  50535.81    0.22          0.00
75%         153694.50   9786.00    66372.00     11.35    346.40  58314.70    0.27          0.00
max         185503.50  10147.00    69509.00     15.80    539.90  70923.94    0.31      60810.00
CV%             18.2%     6.7%        5.1%     41.7%     27.8%     18.5%   28.2%        399.3%
```

---

## 9. Next Steps

Based on this analysis, you can now proceed with:

1. **No changes needed** for basic preprocessing - current pipeline is sound
2. **Consider** using fs_consensus instead of fs_linear for better accuracy
3. **Consider** adding additional feature engineering (rolling statistics)
4. **Re-run** with fixed LSTM issues (batch_size and tensor shape)

---

*Report generated from analysis of run_20260129_1427 outputs*
