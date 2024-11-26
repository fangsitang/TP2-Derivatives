import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression


### ------------ PART ONE ------------
### 1)

# Parameters
S0 = 100  # Spot price
T = 21 / 252  # Time to maturity
r = 0.05  # Risk-free rate
y = 0.02  # Dividend yield
v0 = 0.25  # Initial variance
kappa = 0.2  # Mean reversion speed
theta = 0.2  # Long-run variance
sigma = 0.3  # Volatility of variance
rho = -0.2  # Correlation
strikes = [95, 100, 105]  # Strike prices

# Characteristic function of log(S_t)
def heston_characteristic_function(phi, kappa, theta, sigma, rho, v0, S0, T, r, y):
    u = -0.5
    b = kappa + rho * sigma * phi * 1j
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j))
    g = (b - d) / (b + d)
    C = r * phi * 1j * T + (kappa * theta / sigma**2) * (
        (b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((b - d) / sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    return np.exp(C + D * v0 + phi * 1j * np.log(S0 * np.exp(-y * T)))

# Heston call price integrand
def heston_integrand(phi, K, kappa, theta, sigma, rho, v0, S0, T, r, y):
    cf = heston_characteristic_function(phi - 1j, kappa, theta, sigma, rho, v0, S0, T, r, y)
    return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

# Heston call price
def heston_call_price(K, kappa, theta, sigma, rho, v0, S0, T, r, y):
    integral, _ = quad(lambda phi: heston_integrand(phi, K, kappa, theta, sigma, rho, v0, S0, T, r, y), 0, 100)
    return S0 * np.exp(-y * T) - np.sqrt(K) * np.exp(-r * T) * integral / np.pi

# Compute prices for given strikes
call_prices = [heston_call_price(K, kappa, theta, sigma, rho, v0, S0, T, r, y) for K in strikes]

# Print results
for K, price in zip(strikes, call_prices):
    print(f"Strike {K}: Call Price = {price:.4f}")


### 2) ------------------------


H = 90  # Barrier level
n_paths = 100000  # Number of Monte Carlo paths
n_steps = int(T * 252)  # Number of time steps
dt = 1 / 252  # Time step

# Seed for reproducibility
np.random.seed(42)

# Simulate Heston dynamics
def simulate_heston(S0, v0, kappa, theta, sigma, rho, r, y, T, n_paths, n_steps, dt, H):
    S = np.zeros((n_paths, n_steps + 1))  # Stock prices
    v = np.zeros((n_paths, n_steps + 1))  # Variances
    S[:, 0] = S0
    v[:, 0] = v0
    barrier_breached = np.zeros(n_paths, dtype=bool)  # Track barrier breach

    # Generate correlated Brownian motions
    Z1 = np.random.normal(size=(n_paths, n_steps))
    Z2 = np.random.normal(size=(n_paths, n_steps))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    for t in range(1, n_steps + 1):
        v[:, t] = np.abs(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt +
                         sigma * np.sqrt(v[:, t - 1] * dt) * W2[:, t - 1])
        S[:, t] = S[:, t - 1] * np.exp((r - y - 0.5 * v[:, t - 1]) * dt +
                                       np.sqrt(v[:, t - 1] * dt) * W1[:, t - 1])
        barrier_breached = barrier_breached | (S[:, t] <= H)

    return S, barrier_breached

# Monte Carlo pricing of down-and-out call options
def monte_carlo_down_and_out(S0, v0, kappa, theta, sigma, rho, r, y, T, strikes, n_paths, n_steps, dt, H):
    S, barrier_breached = simulate_heston(S0, v0, kappa, theta, sigma, rho, r, y, T, n_paths, n_steps, dt, H)
    call_prices = []
    S_T = S[:, -1]  # Terminal stock prices

    for K in strikes:
        payoffs = np.where(~barrier_breached, np.maximum(S_T - K, 0), 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        call_prices.append(price)

    return call_prices

# Compute prices for given strikes
down_and_out_prices = monte_carlo_down_and_out(S0, v0, kappa, theta, sigma, rho, r, y, T, strikes, n_paths, n_steps, dt, H)

# Print results
for K, price in zip(strikes, down_and_out_prices):
    print(f"Strike {K}: Down-and-Out Call Price = {price:.4f}")


### 3) ------------------------

# Vanilla European call pricing using Monte Carlo
def monte_carlo_vanilla_call(S, K, r, T):
    S_T = S[:, -1]  # Terminal stock prices
    payoffs = np.maximum(S_T - K, 0)
    return np.exp(-r * T) * payoffs


# Monte Carlo refinement with control variates
def control_variate_refinement(S0, v0, kappa, theta, sigma, rho, r, y, T, strikes, n_paths, n_steps, dt, H, n_batches):
    # Split paths into batches
    batch_size = n_paths // n_batches
    S, barrier_breached = simulate_heston(S0, v0, kappa, theta, sigma, rho, r, y, T, n_paths, n_steps, dt, H)

    # Store regression results
    slopes = []
    r_squared = []

    # For each strike
    for K in strikes:
        down_and_out_values = []
        vanilla_call_values = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Batch data
            S_batch = S[start_idx:end_idx]
            barrier_breached_batch = barrier_breached[start_idx:end_idx]

            # Down-and-out payoffs
            down_and_out_payoffs = np.where(~barrier_breached_batch, np.maximum(S_batch[:, -1] - K, 0), 0)
            down_and_out_values.append(np.exp(-r * T) * down_and_out_payoffs)

            # Vanilla call payoffs
            vanilla_call_payoffs = monte_carlo_vanilla_call(S_batch, K, r, T)
            vanilla_call_values.append(vanilla_call_payoffs)

        # Combine all batches
        down_and_out_values = np.concatenate(down_and_out_values)
        vanilla_call_values = np.concatenate(vanilla_call_values)

        # Perform regression
        reg = LinearRegression()
        X = vanilla_call_values.reshape(-1, 1)
        Y = down_and_out_values
        reg.fit(X, Y)
        beta = reg.coef_[0]
        r2 = reg.score(X, Y)

        slopes.append(beta)
        r_squared.append(r2)

        # Plot data and regression line for one batch
        if K == strikes[0]:  # Example plot for the first strike
            plt.scatter(X, Y, alpha=0.3, label="Data points")
            plt.plot(X, reg.predict(X), color='red', label="Regression line")
            plt.xlabel("Vanilla Call Value")
            plt.ylabel("Down-and-Out Call Value")
            plt.title(f"Regression for Strike {K}")
            plt.legend()
            plt.show()

    return slopes, r_squared


# Run control variate refinement
slopes, r_squared = control_variate_refinement(S0, v0, kappa, theta, sigma, rho, r, y, T, strikes, n_paths, n_steps, dt,
                                               H, n_batches=40)

# Report results
for K, slope, r2 in zip(strikes, slopes, r_squared):
    print(f"Strike {K}: Slope = {slope:.4f}, R^2 = {r2:.4f}")





### ------------ PART TWO ------------
### 1)

# Load data

# Load the data
file_path = "dataTP2_A2024.csv"  # Replace with your file path
data = pd.read_csv(file_path, header=None)  # Assuming no headers in the CSV
prices = data.iloc[:, 0]  # First column with adjusted close prices

# Compute log daily returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Extract adjusted closing prices
closing_prices = data.iloc[:, 0]

# Plot the adjusted closing prices
plt.figure(figsize=(12, 6))
plt.plot(closing_prices, label='Adjusted Close Price', color='blue')
plt.title('Adjusted Closing Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plot the daily returns
plt.figure(figsize=(12, 6))
plt.plot(log_returns, label='Log Returns', color='green')
plt.title('Log Returns')
plt.xlabel('Index')
plt.ylabel('Log Returns')
plt.legend()
plt.grid()
plt.show()

# Compute sample return volatility
annualized_volatility = log_returns.std() * np.sqrt(252)
print(f"Sample Annualized Return Volatility: {annualized_volatility:.2%}")



### 2) ------------------------------------

# Parameters
daily_rf_rate = 0.0275 / 365  # Daily risk-free rate
initial_variance = log_returns.iloc[0] ** 2  # Initialize h_0 with the square of the first return

# NGARCH(1,1) log-likelihood function
def ngarch_log_likelihood(params, returns):
    mu, omega, alpha, beta, gamma, lam = params

    # Initialize variance and arrays
    n = len(returns)
    h = np.zeros(n)
    h[0] = initial_variance
    z = np.zeros(n)

    log_likelihood = 0

    # Recursively compute variances and residuals
    epsilon = 1e-8  # Small value to prevent division by zero
    for t in range(1, n):
        # Update variance
        h[t] = omega + alpha * h[t - 1] * (z[t - 1] - gamma) ** 2 + beta * h[t - 1]
        if h[t] < 1e-8:  # Prevent non-positive variance
            h[t] = 1e-8
        elif h[t] > 1e6:  # Cap excessively large variances
            h[t] = 1e6

        # Update standardized residuals
        z[t] = (returns[t] - daily_rf_rate - lam * np.sqrt(h[t]) + 0.5 * h[t]) / (np.sqrt(h[t]) + epsilon)

        # Increment log-likelihood
        log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) + z[t] ** 2)

    return -log_likelihood  # Negative for minimization


# Initial parameter guesses
initial_guess = [0.0, 0.1, 0.1, 0.8, 0.0, 0.1]  # [mu, omega, alpha, beta, gamma, lambda]
bounds = [
    (-np.inf, np.inf),  # mu can be any real number
    (1e-6, 1),  # omega > 0 and reasonably small
    (1e-6, 0.5),  # 0 < alpha < 0.5
    (1e-6, 0.9),  # 0 < beta < 0.9 (stationarity requires alpha + beta < 1)
    (-1, 1),  # -1 < gamma < 1
    (-5, 5),  # Reasonable bounds for lambda
]


# Optimization
result = minimize(
    ngarch_log_likelihood,
    initial_guess,
    args=(log_returns.values,),
    bounds=bounds,
    method="L-BFGS-B"
)

# Extract estimated parameters
mu, omega, alpha, beta, gamma, lam = result.x

# Compute implied unconditional volatility
unconditional_variance = omega / (1 - alpha * (1 + gamma ** 2) - beta)
unconditional_volatility = np.sqrt(unconditional_variance)

# Compare to sample volatility
sample_volatility = log_returns.std()

# Print results
print("Estimated Parameters:")
print(f"mu: {mu}")
print(f"omega: {omega}")
print(f"alpha: {alpha}")
print(f"beta: {beta}")
print(f"gamma: {gamma}")
print(f"lambda: {lam}")
print("\nUnconditional Volatility:")
print(f"Implied: {unconditional_volatility:.6f}")
print(f"Sample: {sample_volatility:.6f}")


### 3) ------------------------------------
np.random.seed(777)  # For reproducibility

# Given parameters
spot_price = 24789.28  # S_0, the spot price of the index
risk_free_rate_annual = 2.75 / 100  # 2.75% annual risk-free rate
expiry_days = 63  # Days to expiry
daily_rf_rate = risk_free_rate_annual / 365  # Convert to daily risk-free rate
sample_volatility = log_returns.std()  # Sample volatility
annualized_volatility = sample_volatility * np.sqrt(252)  # Annualized volatility

# Strike prices (from 23,000 to 28,000 with increments of 100)
strikes = np.arange(23000, 28001, 100)

# Time to maturity in years (63 days)
T = expiry_days / 252


# Black-Scholes formula for call option price
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call price
    call_price = S0 * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
    return call_price


# Calculate call prices for each strike
call_prices = []
for K in strikes:
    call_price = black_scholes_call(spot_price, K, T, daily_rf_rate, annualized_volatility)
    call_prices.append(call_price)

# Store the results in a DataFrame
options_df = pd.DataFrame({
    'Strike': strikes,
    'Option Price': call_prices
})

print(options_df)


import numpy as np
import pandas as pd

# Given parameters:
num_paths = 100000
num_days = 63
strikes = np.arange(23000, 28001, 100)  # Strike prices from 23,000 to 28,000
S_0 = data.iloc[-1, 0]  # Set observed adjusted close as S_0
daily_rf_rate = 0.0275 / 365  # Daily risk-free rate (2.75% annualized)
omega = 0.000006
alpha = 0.12191
beta = 0.771516
gamma = -0.052665
lam = 0.097762
initial_variance = 0.0001  # Initial guess for variance (adjust as necessary)

# Initialize arrays for simulation
simulated_prices = np.zeros((num_paths, num_days))
simulated_prices[:, 0] = S_0  # Start from the initial spot price

# Simulate paths
h_sim = np.full(num_paths, initial_variance)  # Initialize variance for all paths
z_sim = np.zeros(num_paths)  # Residuals for all paths

# Simulate 100,000 paths over 63 days
for t in range(1, num_days):
    # Simulate shocks (z_t)
    z_sim = np.random.normal(0, 1, num_paths)

    # Update variance using NGARCH(1,1) model
    h_sim = omega + alpha * h_sim * (z_sim - gamma)**2 + beta * h_sim
    h_sim = np.maximum(h_sim, 1e-8)  # Prevent non-positive variance

    # Update returns (under risk-neutral measure, only include the risk-free rate)
    r_sim = daily_rf_rate - 0.5 * h_sim + np.sqrt(h_sim) * z_sim

    # Update prices using the risk-neutral return (rf as the drift term)
    simulated_prices[:, t] = simulated_prices[:, t-1] * np.exp(r_sim)

# Evaluate European Call Options at t = 0
discount_factor = np.exp(-daily_rf_rate * num_days)
call_prices_at_t0 = []

# Calculate call prices at t=0
for K in strikes:
    # Compute payoffs for all paths at t = T (at maturity)
    payoffs = np.maximum(simulated_prices[:, -1] - K, 0)

    # Discount and average to get option price at t = 0
    call_price_at_t0 = discount_factor * np.mean(payoffs)
    call_prices_at_t0.append(call_price_at_t0)

# Output results
print("Strike Price | European Call Price")
for strike, price in zip(strikes, call_prices_at_t0):
    print(f"{strike:12} | {price:20.6f}")



### 4) ---------- PLOTTING VOLATILITY SMILE ----------

# Calculate moneyness S_0 / K for each strike price
moneyness = S_0 / strikes

# Calculate implied volatility for each strike using fsolve
implied_vols = []
for i in range(len(strikes)):
    K = strikes[i]
    market_price = call_prices_at_t0[i]

    # Calculate implied volatility
    implied_vol = implied_volatility_call(S_0, K, num_days / 252, daily_rf_rate, market_price)
    implied_vols.append(implied_vol)

# Plot the volatility smile based on moneyness
plt.figure(figsize=(10, 6))
plt.plot(moneyness, implied_vols, marker='o', color='blue', linestyle='-', label="Implied Volatility")

# Add labels and title
plt.title("Volatility Smile")
plt.xlabel("Moneyness - S(0) / K")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


