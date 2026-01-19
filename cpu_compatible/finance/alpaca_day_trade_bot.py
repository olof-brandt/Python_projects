"""
Day Trading Script for NVIDIA Stock using Alpaca API

This script performs a simple day trading strategy:
- It buys 1 share of NVIDIA (NVDA).
- It continuously monitors the stock price every 60 seconds.
- If the stock gains 2% or more since purchase, it sells to realize profit.
- If the stock loses 1% or more since purchase, it sells to limit loss.

Note: For testing purposes, this script uses Alpaca's paper trading API.
Ensure your API keys are correctly set and have sufficient permissions.
"""

import alpaca_trade_api as tradeapi
import time

# Replace with your actual API key and secret key
API_KEY = 'ooo'
API_SECRET = 'ooo'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

# Stock symbol for NVIDIA
SYMBOL = "NVDA"

# Profit target and stop-loss thresholds
PROFIT_TARGET = 0.02  # 2% profit
STOP_LOSS = -0.01     # 1% loss

def day_trade_nvda():
    try:
        # Retrieve the current market price of NVIDIA
        current_trade = api.get_latest_trade(SYMBOL)
        current_price = float(current_trade.price)
        print(f"Current price of {SYMBOL}: ${current_price:.2f}")

        # Place a market order to buy 1 share
        api.submit_order(
            symbol=SYMBOL,
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Bought 1 share of {SYMBOL} at ${current_price:.2f}")

        # Record the purchase price for profit/loss calculations
        buy_price = current_price

        while True:
            # Wait for 60 seconds before checking the price again
            time.sleep(60)

            # Fetch the latest price
            current_trade = api.get_latest_trade(SYMBOL)
            current_price = float(current_trade.price)
            print(f"Current price of {SYMBOL}: ${current_price:.2f}")

            # Calculate percentage change from the buy price
            price_change_percentage = (current_price - buy_price) / buy_price

            # Check if profit target or stop-loss has been reached
            if price_change_percentage >= PROFIT_TARGET:
                # Sell to take profit
                api.submit_order(
                    symbol=SYMBOL,
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                print(f"Sold 1 share of {SYMBOL} at ${current_price:.2f} (Profit target reached)")
                break
            elif price_change_percentage <= STOP_LOSS:
                # Sell to prevent further loss
                api.submit_order(
                    symbol=SYMBOL,
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                print(f"Sold 1 share of {SYMBOL} at ${current_price:.2f} (Stop loss triggered)")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Start the day trading process
    day_trade_nvda()
