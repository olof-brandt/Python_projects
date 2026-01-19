"""
This script simulates electrical sensor data for a train power system over a 60-second period.
It generates data for voltage, current, calculates power and energy, and visualizes these parameters
using a simple line plot.

Key features:
- Simulates sensor readings at 20 ms intervals.
- Creates a pandas DataFrame with timestamped data.
- Plots voltage, current, and power over time for analysis.

Note:
- Replace the simulation with actual sensor data collection in a real-world scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random  # For simulating data; replace with actual sensor readings in production

# Constant value: interval between sensor readings in milliseconds
SENSOR_READING_INTERVAL_MS = 20


def simulate_data():
    """
    Simulates sensor data for 60 seconds at specified intervals.
    Generates voltage (V), current (A), calculates power (W), and energy (kWh).

    Returns:
        pd.DataFrame: DataFrame containing timestamp, voltage, current, power, and energy.
    """
    start_time = datetime.now()
    voltage_data = []
    current_data = []
    power_data = []
    energy_data = []

    current_time = start_time
    end_time = start_time + timedelta(seconds=60)

    while current_time <= end_time:
        # Simulate voltage in volts (typical for overhead lines)
        voltage = random.uniform(1500, 3000)  # Example range
        # Simulate current in amperes (typical for trains)
        current = random.uniform(0, 500)  # Example range
        # Calculate power in watts (W = V * I)
        power = voltage * current
        # Calculate energy in watt-hours (Wh) for the current interval
        energy = power * (SENSOR_READING_INTERVAL_MS / 1000)  # Wh

        voltage_data.append(voltage)
        current_data.append(current)
        power_data.append(power)
        energy_data.append(energy)

        # Increment time by the sampling interval
        current_time += timedelta(milliseconds=SENSOR_READING_INTERVAL_MS)

    # Create a DataFrame with timestamp and sensor data
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start=start_time, periods=len(voltage_data), freq='20ms'),
        'Voltage (V)': voltage_data,
        'Current (A)': current_data,
        'Power (kW)': np.array(power_data) / 1000,  # Convert W to kW
        'Energy (kWh)': np.array(energy_data) / 3600  # Convert Wh to kWh
    })

    return data


def create_dashboard(data):
    """
    Creates a line plot dashboard for voltage, current, and power over time.

    Args:
        data (pd.DataFrame): DataFrame containing sensor data with timestamps.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(data['Timestamp'], data['Voltage (V)'], label='Voltage (V)')
    plt.plot(data['Timestamp'], data['Current (A)'], label='Current (A)')
    plt.plot(data['Timestamp'], data['Power (kW)'], label='Power (kW)')

    plt.xlabel('Time')
    plt.ylabel('Sensor Readings')
    plt.title('Electrical Parameters Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main execution: simulate data and create visualization
if __name__ == "__main__":
    data = simulate_data()
    create_dashboard(data)
