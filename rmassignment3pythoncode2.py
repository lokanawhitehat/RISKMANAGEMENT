import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import matplotlib.pyplot as plt

def download_bitcoin_data(start_date, end_date):
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    if btc.empty:
        print("Error: No data downloaded.")
        return None
    print("Available columns:\n", btc.columns)  # Debugging statement
    btc = btc['Close'] if isinstance(btc, pd.DataFrame) and 'Close' in btc.columns else btc  # Extract 'Close' price properly
    btc = btc.to_frame() if isinstance(btc, pd.Series) else btc  # Ensures btc remains a DataFrame
    btc.columns = ['Close']
    btc['Returns'] = btc['Close'].pct_change()
    btc.dropna(inplace=True)
    return btc

def ewma_volatility(data, lambda_):
    if data.empty:
        print("Error: Empty dataset provided to EWMA calculation.")
        return data
    var_t = np.zeros(len(data))
    var_t[0] = np.var(data['Returns']) if len(data) > 0 else 0
    for t in range(1, len(data)):
        var_t[t] = lambda_ * var_t[t-1] + (1 - lambda_) * (data['Returns'].iloc[t-1] ** 2)
    data['Volatility'] = np.sqrt(var_t)
    return data

def calculate_var(data, confidence=0.95):
    if 'Volatility' not in data:
        print("Error: Volatility column missing in dataset.")
        return data
    z_score = stats.norm.ppf(1 - confidence)
    data['VaR'] = -z_score * data['Volatility']
    return data

def backtest_var(data):
    if data.empty or 'VaR' not in data or 'Returns' not in data:
        print("Error: Invalid dataset for backtesting.")
        return 0, np.nan
    exceptions = (data['Returns'] < data['VaR']).sum()
    expected_exceptions = len(data) * 0.05
    if exceptions == 0 or exceptions == len(data) or exceptions / len(data) in [0, 1]:
        print("Warning: No exceptions or all values exceeded VaR, returning NaN")
        return exceptions, np.nan  # Return NaN instead of computing
    kupiec_pof = -2 * np.log(((1 - 0.05)**(len(data) - exceptions) * (0.05 ** exceptions)) /
                             (((1 - (exceptions / len(data))) ** (len(data) - exceptions)) * ((exceptions / len(data)) ** exceptions)))
    p_value_pof = 1 - stats.chi2.cdf(kupiec_pof, df=1)
    return exceptions, p_value_pof

def ten_day_risk_measures(volatility, confidence=0.95):
    if volatility is None or np.isnan(volatility):
        print("Error: Invalid volatility input for risk measures.")
        return np.nan, np.nan, np.nan
    z_score = stats.norm.ppf(1 - confidence)
    phi = stats.norm.pdf(z_score)
    ten_day_vol = volatility * np.sqrt(10)
    var_10 = -z_score * ten_day_vol
    etl_10 = -phi / (1 - confidence) * ten_day_vol
    max_var_10 = -z_score * ten_day_vol
    return var_10, etl_10, max_var_10

# Download Bitcoin data from 2015 to 2025
start_date, end_date = '2015-01-01', '2025-02-01'
btc_data = download_bitcoin_data(start_date, end_date)

if btc_data is not None and not btc_data.empty:
    # Compute EWMA volatility for different lambda values
    for lambda_ in [0.94, 0.97, 0.70]:
        btc_data = ewma_volatility(btc_data, lambda_)
        btc_data = calculate_var(btc_data)
        btc_subset = btc_data.loc['2020-01-01':] if '2020-01-01' in btc_data.index else btc_data
        exceptions, p_value_pof = backtest_var(btc_subset)
        print(f'Lambda: {lambda_}, Exceptions: {exceptions}, Kupiec POF p-value: {p_value_pof}')

    # Compute 10-day risk measures for 02/01/2025 using EWMA(0.94) volatility
    if 'Volatility' in btc_data:
        volatility_2025 = btc_data.loc['2025-01-31', 'Volatility'] if '2025-01-31' in btc_data.index else None
        var_10, etl_10, max_var_10 = ten_day_risk_measures(volatility_2025)
        print(f'10-Day 95% VaR: {var_10}, 10-Day 95% ETL: {etl_10}, 10-Day 95% MaxVaR: {max_var_10}')








