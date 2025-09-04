import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def straddle_price(an_vol, time=1):
    """
    Computes the ATMF straddle price approximation.
    
    Parameters:
        an_vol (float): Annualized volatility.
        time (float): Time to maturity in years.

    Returns:
        float: Estimated straddle price.
    """
    return (2 / (2 * np.pi) ** 0.5) * an_vol * np.sqrt(time)


def straddle_pricer_mc(an_vol=0.2, time=1, mc_paths=100):
    """
    Monte Carlo simulation to price a straddle.

    Parameters:
        an_vol (float): Annualized volatility.
        time (float): Time in years.
        mc_paths (int): Number of Monte Carlo simulations.

    Returns:
        float: Estimated straddle price.
    """
    daily_vol = an_vol / np.sqrt(252)
    n_days = int(time * 252)
    result_sum = 0

    for _ in range(mc_paths):
        returns = np.random.normal(0, daily_vol, n_days)
        result_sum += np.abs(np.prod(1 + returns) - 1)

    return result_sum / mc_paths


def five_prices(an_vol=0.2, time=1, mc_paths=10000):
    """
    Generate 5 straddle price estimates using vectorized simulation.

    Returns:
        list[float]: List of 5 estimated straddle prices.
    """
    daily_vol = an_vol / np.sqrt(252)
    n_days = int(time * 252)
    results = []

    for _ in range(5):
        data = np.random.normal(0, daily_vol, (n_days, mc_paths))
        price = (np.abs((data + 1).prod(axis=0) - 1).sum()) / mc_paths
        results.append(float(price))

    return results


def simulate_asset_path(an_vol=0.2, time=1):
    """
    Simulates a single asset's path and calculates straddle return.

    Returns:
        pd.DataFrame: Simulated asset price path.
        float: Straddle return.
    """
    daily_vol = an_vol / np.sqrt(252)
    n_days = int(time * 252)
    returns = np.random.normal(0, daily_vol, n_days)
    straddle_return = np.abs(np.prod(1 + returns) - 1)

    df = pd.DataFrame(returns + 1, columns=['return'])
    df['price'] = df['return'].cumprod()

    return df, straddle_return


# --- Run code ---

# 1. Basic straddle price
print(f"Analytical straddle price: {straddle_price(0.2):.4f}")

# 2. Monte Carlo simulation (simple)
print(f"MC estimated straddle price: {straddle_pricer_mc():.4f}")

# 3. Five straddle price estimates, demonstrating reduced variance.
print(f"Five estimated prices: {five_prices()}")

# 4. Simulated path + straddle return
simulated_asset, straddle_return = simulate_asset_path()
print(f"Straddle return from single path: {straddle_return:.4f}")
print(simulated_asset.head())

# 5. Monte Carlo average using fixed return
mc_paths = 100
mc_avg = np.mean([straddle_return for _ in range(mc_paths)])
print(f"MC average using fixed return: {mc_avg:.4f}")

# 6. Simulation of multiple paths
n_days = int(252)
simulated_matrix = pd.DataFrame(
    np.random.normal(0, 0.2 / np.sqrt(252), (n_days, 4))
)
simulated_prices = (1 + simulated_matrix).cumprod()

# Visualisation of straddle paths
plt.style.use('tradesoc')  # Custom style
plt.figure(figsize=(8, 6))
plt.plot(simulated_prices)
plt.xlabel('Days')
plt.ylabel('Straddle Price Multiplier')
plt.title('Simulated Straddle Paths')
plt.grid(True)
plt.tight_layout()
plt.show()


