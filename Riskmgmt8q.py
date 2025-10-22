from math import log, sqrt, exp
from scipy.stats import norm

# Define functions for Black-Scholes components
def d1(S, K, r, T, sigma):
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - sigma * sqrt(T)

# Option price (Call)
def option_price(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    return S * norm.cdf(d1_val) - K * exp(-r * T) * norm.cdf(d2_val)

# Greeks calculations
def delta(S, K, r, T, sigma):
    return norm.cdf(d1(S, K, r, T, sigma))

def gamma(S, K, r, T, sigma):
    return norm.pdf(d1(S, K, r, T, sigma)) / (S * sigma * sqrt(T))

def vega(S, K, r, T, sigma):
    return S * norm.pdf(d1(S, K, r, T, sigma)) * sqrt(T)

def theta(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    first_term = -(S * norm.pdf(d1_val) * sigma) / (2 * sqrt(T))
    second_term = -r * K * exp(-r * T) * norm.cdf(d2_val)
    return first_term + second_term

def rho(S, K, r, T, sigma):
    d2_val = d2(S, K, r, T, sigma)
    return K * T * exp(-r * T) * norm.cdf(d2_val)

# Given inputs
S = 30  # Stock price
K = 30  # Strike price
r = 0.05  # Risk-free rate
T = 1  # Time to maturity (in years)
sigma = 0.25  # Volatility

# Calculate option price and Greeks
price = option_price(S, K, r, T, sigma)
delta_value = delta(S, K, r, T, sigma)
gamma_value = gamma(S, K, r, T, sigma)
vega_value = vega(S, K, r, T, sigma)
theta_value = theta(S, K, r, T, sigma)
rho_value = rho(S, K, r, T, sigma)

# Display initial results
print(f"Option Price: {price}")
print(f"Delta: {delta_value}")
print(f"Gamma: {gamma_value}")
print(f"Vega: {vega_value}")
print(f"Theta: {theta_value}")
print(f"Rho: {rho_value}")

# Verifications
# Verify Delta
new_price = option_price(30.1, K, r, T, sigma)
delta_check = (new_price - price) / (30.1 - S)
print(f"Verified Delta: {delta_check}")

# Verify Gamma
new_delta = delta(30.1, K, r, T, sigma)
gamma_check = (new_delta - delta_value) / (30.1 - S)
print(f"Verified Gamma: {gamma_check}")

# Verify Vega
new_price_vega = option_price(S, K, r, T, 0.26)
vega_check = (new_price_vega - price) / (0.26 - sigma)
print(f"Verified Vega: {vega_check}")

# Verify Theta
new_price_theta = option_price(S, K, r, T - 1/365, sigma)
theta_check = (new_price_theta - price) / (-1/365)
print(f"Verified Theta: {theta_check}")

# Verify Rho
new_price_rho = option_price(S, K, 0.051, T, sigma)
rho_check = (new_price_rho - price) / (0.051 - r)
print(f"Verified Rho: {rho_check}")
