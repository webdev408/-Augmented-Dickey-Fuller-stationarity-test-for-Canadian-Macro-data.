# Augmented Dickey-Fuller Stationarity Test for Canadian Macro Data

This repository provides a comprehensive illustration of the Augmented Dickey-Fuller (ADF) method for testing stationarity in time series data, applied to Canadian macroeconomic variables.

## Overview

This analysis demonstrates a complete workflow for testing stationarity in macroeconomic time series:

1. **Level Regression Analysis** - Initial regression with variables in their original levels
2. **ADF Stationarity Test (Levels)** - Testing for unit roots in level variables
3. **First Difference Model** - Regression with first-differenced variables
4. **ADF Stationarity Test (First Differences)** - Confirming stationarity after differencing

## Dataset

The analysis uses Canadian macroeconomic data from 1990-2020, including:

- **GDP** (Target Variable): Gross Domestic Product (billions CAD)
- **population**: Population in millions
- **longevity**: Life expectancy in years
- **mean_taxRate**: Average tax rate (%)

## Key Findings

### Level Variables (Non-Stationary)
The ADF test reveals that most variables at their levels are **non-stationary**, indicating the presence of unit roots and trends in the data. This is typical for macroeconomic time series.

### First Differences (Stationary)
After taking first differences, the variables become **stationary**, confirming that they are integrated of order 1, I(1). This transformation removes the trend component and makes the data suitable for time series modeling.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0

## Usage

Run the complete analysis:

```bash
python adf_analysis.py
```

This will:
1. Load the Canadian macroeconomic data
2. Perform level regression analysis
3. Conduct ADF tests on level variables
4. Fit a first difference model
5. Conduct ADF tests on first differences
6. Generate visualizations saved as `canadian_macro_analysis.png`

## Output

The script provides:
- Detailed ADF test statistics for each variable
- OLS regression results for both level and first difference models
- Summary tables comparing stationarity across variables
- Visual plots showing both level and differenced time series
- Comprehensive interpretation of results

## Statistical Methodology

### Augmented Dickey-Fuller Test
The ADF test checks the null hypothesis that a unit root is present in the time series:
- **H₀**: Unit root exists (non-stationary)
- **H₁**: No unit root (stationary)

A p-value < 0.05 rejects the null hypothesis, indicating the series is stationary.

### First Differencing
First differencing is calculated as: Δyₜ = yₜ - yₜ₋₁

This transformation is commonly used to achieve stationarity in I(1) time series.

## Files

- `adf_analysis.py` - Main analysis script
- `canadian_macro_data.csv` - Dataset with Canadian macroeconomic variables
- `requirements.txt` - Python package dependencies
- `canadian_macro_analysis.png` - Generated visualization (after running script)

## References

- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366a), 427-431.
- Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. Biometrika, 71(3), 599-607.

## License

This project is open source and available for educational purposes.
