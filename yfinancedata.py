import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]

# Download data
data = yf.download(tickers, start="2018-01-01", end="2025-02-01")

# Print the first few rows
print(data.head())

# Save to CSV
data.to_csv("dow30_prices.csv")

