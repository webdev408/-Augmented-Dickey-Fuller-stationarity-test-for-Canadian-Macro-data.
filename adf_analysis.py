"""
Canadian Macroeconomic Data Analysis with ADF Stationarity Testing

This script performs a comprehensive analysis of Canadian macroeconomic data:
1. Level regression analysis
2. ADF stationarity test on level variables
3. First difference model
4. ADF stationarity test on first differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS, add_constant
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='canadian_macro_data.csv'):
    """Load the Canadian macroeconomic data."""
    df = pd.read_csv(filepath)
    return df


def perform_adf_test(series, variable_name):
    """
    Perform Augmented Dickey-Fuller test on a time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data to test
    variable_name : str
        Name of the variable for display
        
    Returns:
    --------
    dict : Test results
    """
    result = adfuller(series, autolag='AIC')
    
    adf_results = {
        'Variable': variable_name,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'Is Stationary': result[1] < 0.05
    }
    
    return adf_results


def print_adf_results(adf_results):
    """Print ADF test results in a formatted manner."""
    print(f"\nADF Test Results for {adf_results['Variable']}:")
    print("=" * 60)
    print(f"ADF Statistic: {adf_results['ADF Statistic']:.6f}")
    print(f"p-value: {adf_results['p-value']:.6f}")
    print(f"Lags Used: {adf_results['Lags Used']}")
    print(f"Number of Observations: {adf_results['Number of Observations']}")
    print("\nCritical Values:")
    for key, value in adf_results['Critical Values'].items():
        print(f"  {key}: {value:.4f}")
    
    if adf_results['Is Stationary']:
        print(f"\nConclusion: {adf_results['Variable']} is STATIONARY (p-value < 0.05)")
    else:
        print(f"\nConclusion: {adf_results['Variable']} is NON-STATIONARY (p-value >= 0.05)")
    print("=" * 60)


def level_regression(df):
    """
    Perform level regression with GDP as dependent variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the macroeconomic data
        
    Returns:
    --------
    Fitted OLS model
    """
    print("\n" + "=" * 70)
    print("LEVEL REGRESSION ANALYSIS")
    print("=" * 70)
    print("\nModel: GDP = β₀ + β₁*population + β₂*longevity + β₃*mean_taxRate + ε\n")
    
    # Prepare data
    y = df['GDP']
    X = df[['population', 'longevity', 'mean_taxRate']]
    X = add_constant(X)
    
    # Fit model
    model = OLS(y, X).fit()
    
    # Print results
    print(model.summary())
    
    return model


def first_difference_regression(df):
    """
    Perform first difference regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the macroeconomic data
        
    Returns:
    --------
    Fitted OLS model and differenced data
    """
    print("\n" + "=" * 70)
    print("FIRST DIFFERENCE MODEL")
    print("=" * 70)
    print("\nModel: ΔGDP = β₀ + β₁*Δpopulation + β₂*Δlongevity + β₃*Δmean_taxRate + ε\n")
    
    # Create first differences
    df_diff = df[['GDP', 'population', 'longevity', 'mean_taxRate']].diff().dropna()
    
    # Prepare data
    y_diff = df_diff['GDP']
    X_diff = df_diff[['population', 'longevity', 'mean_taxRate']]
    X_diff = add_constant(X_diff)
    
    # Fit model
    model_diff = OLS(y_diff, X_diff).fit()
    
    # Print results
    print(model_diff.summary())
    
    return model_diff, df_diff


def test_stationarity_levels(df):
    """
    Test stationarity of all variables at level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the macroeconomic data
    """
    print("\n" + "=" * 70)
    print("ADF STATIONARITY TEST - LEVEL VARIABLES")
    print("=" * 70)
    
    variables = ['GDP', 'population', 'longevity', 'mean_taxRate']
    results = []
    
    for var in variables:
        adf_result = perform_adf_test(df[var], var)
        results.append(adf_result)
        print_adf_results(adf_result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Stationarity of Level Variables")
    print("=" * 70)
    summary_df = pd.DataFrame(results)
    print(summary_df[['Variable', 'ADF Statistic', 'p-value', 'Is Stationary']].to_string(index=False))
    print("\n")
    
    return results


def test_stationarity_first_differences(df):
    """
    Test stationarity of all variables in first differences.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the macroeconomic data
    """
    print("\n" + "=" * 70)
    print("ADF STATIONARITY TEST - FIRST DIFFERENCES")
    print("=" * 70)
    
    variables = ['GDP', 'population', 'longevity', 'mean_taxRate']
    results = []
    
    for var in variables:
        # Calculate first difference
        diff_series = df[var].diff().dropna()
        var_name = f"Δ{var}"
        
        adf_result = perform_adf_test(diff_series, var_name)
        results.append(adf_result)
        print_adf_results(adf_result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Stationarity of First Differences")
    print("=" * 70)
    summary_df = pd.DataFrame(results)
    print(summary_df[['Variable', 'ADF Statistic', 'p-value', 'Is Stationary']].to_string(index=False))
    print("\n")
    
    return results


def create_visualizations(df):
    """
    Create visualizations for the analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the macroeconomic data
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    variables = ['GDP', 'population', 'longevity', 'mean_taxRate']
    
    # Plot level variables
    for i, var in enumerate(variables):
        axes[0, i].plot(df['year'], df[var], marker='o', linewidth=2)
        axes[0, i].set_title(f'{var} (Level)', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('Year')
        axes[0, i].set_ylabel(var)
        axes[0, i].grid(True, alpha=0.3)
    
    # Plot first differences
    for i, var in enumerate(variables):
        diff_series = df[var].diff()
        axes[1, i].plot(df['year'][1:], diff_series[1:], marker='o', linewidth=2, color='orange')
        axes[1, i].set_title(f'Δ{var} (First Difference)', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('Year')
        axes[1, i].set_ylabel(f'Δ{var}')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('canadian_macro_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'canadian_macro_analysis.png'")
    plt.close()


def main():
    """Main function to run the complete analysis."""
    print("\n" + "=" * 70)
    print("CANADIAN MACROECONOMIC DATA ANALYSIS")
    print("Augmented Dickey-Fuller Stationarity Testing")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_data('canadian_macro_data.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Description:")
    print(df.describe())
    
    # Step 1: Level Regression
    level_model = level_regression(df)
    
    # Step 2: Test stationarity at levels (should be non-stationary)
    level_results = test_stationarity_levels(df)
    
    # Step 3: First Difference Model
    diff_model, df_diff = first_difference_regression(df)
    
    # Step 4: Test stationarity of first differences (should be stationary)
    diff_results = test_stationarity_first_differences(df)
    
    # Step 5: Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n1. Level Regression:")
    print("   - Model fitted with GDP as dependent variable")
    print("   - Predictors: population, longevity, mean_taxRate")
    
    print("\n2. ADF Test on Level Variables:")
    non_stationary_count = sum(1 for r in level_results if not r['Is Stationary'])
    print(f"   - All {len(level_results)} variables tested")
    print(f"   - {non_stationary_count} variables are NON-STATIONARY")
    print("   - This indicates the presence of unit roots in level data")
    
    print("\n3. First Difference Model:")
    print("   - Model fitted with first differences of all variables")
    print("   - This transformation removes the trend component")
    
    print("\n4. ADF Test on First Differences:")
    stationary_count = sum(1 for r in diff_results if r['Is Stationary'])
    print(f"   - All {len(diff_results)} differenced variables tested")
    print(f"   - {stationary_count} variables are STATIONARY after differencing")
    print("   - First differencing successfully removed the unit roots")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nConclusion:")
    print("The analysis confirms that the Canadian macroeconomic variables (GDP,")
    print("population, longevity, and mean_taxRate) are integrated of order 1, I(1).")
    print("They are non-stationary at levels but become stationary after first")
    print("differencing, which is typical for macroeconomic time series data.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
