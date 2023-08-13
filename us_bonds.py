import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Define the bond ticker symbol (e.g., 10-year U.S. Treasury)
bond_symbol = "^TNX"

# Fetch bond data
bond_data = yf.Ticker(bond_symbol)


print(bond_data.info)

# Get historical data
bond_history = bond_data.history(period="20y")

print(bond_history)


# Plotting the bond yield over time
plt.figure(figsize=(10, 6))
plt.plot(bond_history.index, bond_history['Close'], label="Bond Yield")
plt.xlabel("Date")
plt.ylabel("Yield")
plt.title("U.S. Bond Yield Over Time")
plt.legend()
plt.grid(True)
plt.show()

print(bond_history.columns)


# Calculate bond duration
def calculate_duration(cash_flows, discount_rates):
    weighted_cash_flows = cash_flows * \
        np.exp(-discount_rates * bond_history.index.year)
    present_value = weighted_cash_flows.sum()
    return present_value / bond_history['Close'].values[-1]


# Assuming cash flows are constant and annual
coupon = bond_data.info['couponRate']  # Replace with actual coupon rate
maturity = bond_data.info['maturityDate']  # Replace with actual maturity date
cash_flows = [coupon] * (maturity.year - bond_history.index.year[0] + 1)
# Assuming bond prices are given in percentage terms
discount_rates = bond_history['Close'] / 100

bond_duration = calculate_duration(
    np.array(cash_flows), np.array(discount_rates))

print("Bond Duration:", bond_duration)
