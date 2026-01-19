"""
This script fetches historical stock data for Apple (AAPL) from Yahoo Finance for the year 2024.
It then displays the first few rows of data and plots the closing prices over time,
including titles and axis labels for clarity.

Key improvements:
- Added descriptive comments explaining each step.
- Set the figure size for better visualization.
- Included plot titles and axis labels.
- Displayed the first five rows of the dataset for inspection.
"""

# Import yfinance to retrieve stock data
import yfinance as yf

# Download historical stock data for Apple (AAPL) from January 1, 2024 to January 1, 2025
stock_data = yf.download("AAPL", start="2024-01-01", end="2025-01-01")

# Display the first 5 rows of the data to understand its structure
print(stock_data.head())

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Create a figure with a specified size (10x6 inches)
plt.figure(figsize=(10, 6))

# Plot the closing prices over time
plt.plot(stock_data.index, stock_data['Close'], label='Closing Price', color='blue')

# Add a legend to identify the plotted line
plt.legend()

# Add grid lines for better readability
plt.grid(True)

# Set the title of the plot
plt.title('Apple Stock Closing Price from 2024-01-01 to 2025-01-01')

# Label the x-axis
plt.xlabel('Date')

# Label the y-axis
plt.ylabel('Closing Price (USD)')

# Display the plot
plt.show()
