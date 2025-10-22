import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of stocks
tickers = ['AAPL', 'CSCO', 'TRV']

# Download historical stock prices
df = yf.download(tickers, start='2020-01-01', end='2023-12-31')['Close']

# Check if data is available
print("Downloaded Data:\n", df.head())

# Calculate daily returns
daily_returns = df.pct_change().dropna()

# Compute correlation matrix
correlation_matrix = daily_returns.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Check available tickers in correlation matrix
print("\nAvailable tickers in correlation matrix:", correlation_matrix.index.tolist())

# Compute correlation between CSCO and TRV (Handle missing data)
if 'CSCO' in correlation_matrix.index and 'TRV' in correlation_matrix.index:
    csco_trv_correlation = correlation_matrix.loc['CSCO', 'TRV']
    print(f"\nCorrelation between CSCO and TRV: {csco_trv_correlation:.4f}")
else:
    print("\nCSCO or TRV data is missing in the correlation matrix!")

# 📌 FIX: Matplotlib Heatmap Issue
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

# Label axes
plt.xticks(range(len(tickers)), tickers, rotation=45)
plt.yticks(range(len(tickers)), tickers)
plt.title("Stock Correlation Heatmap")

# Show the plot
plt.show()


# Resample to annual returns
annual_returns = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

# Annualized standard deviations
annualized_std_devs = daily_returns.resample('YE').std() * np.sqrt(252)

# Create Risk-Return Scatter Plot
plt.figure(figsize=(10, 6))
for ticker in tickers:
    plt.scatter(annualized_std_devs[ticker], annual_returns[ticker], label=ticker)
    for i in range(len(annual_returns)):
        plt.text(annualized_std_devs[ticker].iloc[i], annual_returns[ticker].iloc[i], annual_returns.index.year[i], fontsize=9)

plt.title('Risk-Return Space for CSCO and TRV (2019-2023)')
plt.xlabel('Annualized Standard Deviation of Returns')
plt.ylabel('Annual Return')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

for ticker in tickers:
    plt.hist(daily_returns[ticker], bins=30, alpha=0.5, label=ticker)

plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

for ticker in tickers:
    mean_return = np.mean(daily_returns[ticker])
    std_dev_return = np.std(daily_returns[ticker])
    print(f"{ticker}: Mean Return = {mean_return:.6f}, Std Dev = {std_dev_return:.6f}")


