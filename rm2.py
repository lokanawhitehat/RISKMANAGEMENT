import yfinance as yf
from pandas_datareader import data as pdr

# Override pandas_datareader's Yahoo Finance with yfinance
yf.pdr_override()

# Now, fetching data works
df = pdr.get_data_yahoo('AAPL', start='2020-01-01', end='2023-12-31')

print(df.head())
