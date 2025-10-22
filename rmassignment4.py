import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import yfinance as yf

# Load historical data (try different encodings if needed)
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding="latin1")
    
    # Debugging: Print dataset info
    print("Loaded Data Info:")
    print(data.info())

    # Ensure the index is a date
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Warning: The index is not a recognized datetime. Attempting to convert...")
        data.index = pd.to_datetime(data.index, errors='coerce')
    
    # Convert all data to numeric, forcing errors to NaN and dropping non-numeric columns
    data = data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    return data

# Download stock price data from Yahoo Finance
def download_stock_data(tickers, start_date, end_date, file_name):
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Check if "Adj Close" exists, otherwise fallback to "Close"
    if "Adj Close" in data:
        data = data["Adj Close"]
    else:
        print("Warning: 'Adj Close' not found, using Close prices instead.")
        data = data["Close"]
    
    data.to_csv(file_name)
    print(f"Stock price data saved to {file_name}")

# Define stock tickers for the portfolio
tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
download_stock_data(tickers, "2018-01-01", "2025-02-01", "dow30_prices.csv")

# Load and check data
data = load_data("dow30_prices.csv")
print("First few rows of cleaned data:")
print(data.head())

# Compute returns and ensure numeric values
returns = data.pct_change().dropna()
returns = returns.apply(pd.to_numeric, errors='coerce')
returns = returns.dropna()

# Debugging: Ensure returns are not empty
if returns.empty:
    raise ValueError("Error: Processed returns dataframe is empty. Check the CSV data format!")

# Mean and covariance of asset returns
mu = returns.mean().values  # Convert to numpy array
cov_matrix = returns.cov().values  # Convert to numpy array

# Debugging: Ensure there are assets before optimization
if mu.size == 0 or cov_matrix.size == 0:
    raise ValueError("Error: mu or covariance matrix is empty! Check the data.")

# Ensure numerical stability of covariance matrix
cov_matrix += np.eye(len(mu)) * 1e-6

# Number of assets
n = len(mu)
if n == 0:
    raise ValueError("Error: No valid assets found in the dataset!")

w = cp.Variable(n)
t = cp.Variable()  # Auxiliary variable for DCP-compliant objective
z = cp.Variable()  # Scaling factor for Sharpe ratio
proxy = cp.Variable()  # Proxy variable to ensure DCP compliance
gamma = cp.Variable()  # New transformation variable

# Portfolio optimization for Max Sharpe Ratio
risk_free_rate = 0.05 / 252  # Convert to daily
port_return = mu @ w  # Convert mu to numpy before matrix multiplication

# Debugging: Print constraint values before solving
print("Constraint Values Before Solving:")
print(f"mu: {mu}")
print(f"cov_matrix:\n{cov_matrix}")
print(f"risk_free_rate: {risk_free_rate}")

# Enforce realistic constraints to prevent unbounded solutions
min_expected_return = 0.0001  # Small threshold for minimum expected return
constraints = [
    cp.sum(w) >= 0.99,  # Forces at least 99% investment
    w <= 0.5,  # No single asset gets more than 50% allocation
    cp.sum(w) == 1, 
    w >= 0, 
    w <= 1,  # Ensures weights are within a valid range
    t >= 1e-2,  # Ensure t is strictly positive to avoid numerical instability
    cp.SOC(t, cp.matmul(cp.sqrt(cov_matrix), w)),
    proxy >= port_return - risk_free_rate - 1e-4,  # Relaxed constraint
    cp.SOC(gamma, cp.vstack([2 * proxy, t - z])),  # Ensuring valid scaling using a second-order cone constraint
    port_return >= min_expected_return  # Prevents unrealistic optimization results
]
objective = cp.Maximize(gamma)  # Maximize gamma instead of proxy
problem = cp.Problem(objective, constraints)

# Debugging: Solve only if the problem is well-defined
try:
    problem.solve(solver=cp.ECOS, verbose=True)  # Use SCS solver for better SOC constraint handling
    optimal_weights = w.value
except Exception as e:
    raise RuntimeError(f"Optimization failed: {e}")

# Debugging: Ensure optimal_weights is not None
if optimal_weights is None or len(optimal_weights) != len(returns.columns):
    raise ValueError("Error: Optimization failed, no valid weights found!")

# Get optimized weights
portfolio_df = pd.DataFrame({"Stock": returns.columns, "Weight": optimal_weights})
portfolio_df = portfolio_df[portfolio_df["Weight"] > 0.01]  # Filter stocks with >1% allocation

# Display portfolio allocation
print("Optimized Portfolio Allocation:")
print(portfolio_df)

# Compute daily portfolio returns
portfolio_returns = (returns @ optimal_weights).dropna()

# Compute 95% Parametric VaR
if portfolio_returns.empty:
    raise ValueError("Error: Portfolio returns are empty! Check the data processing.")

var_95 = -np.percentile(portfolio_returns, 5)
print(f"95% Value at Risk (VaR): {var_95:.4f}")

# Plot cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label="Optimized Portfolio", color="blue")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Portfolio Performance Over Time")
plt.legend()
plt.show()



















