"""
Simulation of a PWM-based inverter with phase voltage and current calculations.

This script simulates a simple inverter control system that uses a triangular carrier
wave and sine reference signals to generate PWM signals for switches. It calculates
the resulting phase voltage and current over time, and plots the signals for analysis.

Key features:
- Generates a sine reference signal and a triangular carrier wave.
- Implements PWM logic to switch transistors based on the comparison.
- Calculates the phase voltage based on switch states.
- Integrates the phase current considering the inductance.
- Plots all relevant signals for visualization.

Note:
- The simulation runs for 1 second.
- The delay introduced simulates switch transition delays.
"""

import math
import matplotlib.pyplot as plt
import time

# --- Initialization ---

# Fundamental frequency of the sine reference (Hz)
frequency = 20000

# Amplitude of the sine reference signal
amp_control = 1

# Amplitude of the triangular carrier wave
amp_triangular = 1

# Initialize the triangular wave parameters
triangular_wave = -amp_triangular
tri_max = amp_triangular
tri_min = -amp_triangular

v_triangular = -amp_triangular

# Delay to simulate switch transition time (seconds)
delay = 0.01 * (1 / frequency)

# Switch states (0 or 100 representing off/on)
switch_a1 = 0
switch_a2 = 0
switch_b1 = 0
switch_b2 = 0

# Modulation index parameters
modulation_factor = 10
sine_freq = 3  # Base sine frequency (Hz)
tri_freq = modulation_factor * sine_freq  # Triangular wave frequency

# Slope of the triangular wave
slope_tri = 2 * tri_max * tri_freq

# Initialize the triangular wave slope
tri_slope = slope_tri
tri_slope_prev = slope_tri

# Data storage for plotting
time_start = time.time()
time_now = 0

time_list = []
v_triangular_list = []
v_sine_pos_list = []
v_sine_neg_list = []
v_phase_list = []
i_phase_list = []

# Phase voltage and current parameters
dc_voltage = 40  # DC supply voltage
phase_current = 0  # Initial phase current
inductance = 10e-3  # Inductance in Henry

# --- Simulation Loop ---

while time_now < 1:
    # Update current simulation time
    time_now = time.time() - time_start

    # Calculate the simulation step based on elapsed time
    step = time_now - time_list[-1] if time_list else 0  # Zero for the first iteration

    # Generate the sine reference signal
    v_control = amp_control * math.sin(2 * math.pi * time_now * sine_freq)
    v_control_neg = -v_control

    # Update the triangular carrier wave
    # Change slope direction at the wave's peaks
    if v_triangular + tri_slope * step > tri_max:
        tri_slope = -slope_tri
        tri_slope_prev = tri_slope
    elif v_triangular + tri_slope * step < tri_min:
        tri_slope = slope_tri
        tri_slope_prev = tri_slope
    else:
        tri_slope = tri_slope_prev

    # Integrate to get the current triangular wave value
    v_triangular += tri_slope * step

    # --- PWM Logic: Compare sine reference with carrier wave to determine switch states ---

    # Switch A control
    if v_control > v_triangular:
        switch_a2 = 0
        time.sleep(delay)  # Simulate switch delay
        switch_a1 = 100
    elif v_control < v_triangular:
        switch_a1 = 0
        time.sleep(delay)
        switch_a2 = 100

    # Switch B control
    if v_control_neg > v_triangular:
        switch_b2 = 0
        time.sleep(delay)
        switch_b1 = 100
    elif v_control_neg < v_triangular:
        switch_b1 = 0
        time.sleep(delay)
        switch_b2 = 100

    # --- Determine phase voltage based on switch states ---
    if switch_a1 == 100 and switch_b2 == 100:
        v_phase = dc_voltage
    elif switch_b1 == 100 and switch_a2 == 100:
        v_phase = -dc_voltage
    else:
        v_phase = 0

    # Store data for plotting
    time_list.append(time_now)
    v_triangular_list.append(v_triangular)
    v_sine_pos_list.append(v_control)
    v_sine_neg_list.append(v_control_neg)
    v_phase_list.append(v_phase)

    # --- Calculate phase current ---
    didt = v_phase / inductance  # Derivative of current
    phase_current += didt * step  # Numerical integration
    i_phase_list.append(phase_current)

# --- Plotting Results ---

fig, axs = plt.subplots(4, 1, figsize=(10, 8))

axs[0].plot(time_list, v_triangular_list)
axs[0].set_title("Triangular Carrier Wave")
axs[0].set_ylabel("Voltage (V)")

axs[1].plot(time_list, v_sine_pos_list, label="Sine Positive")
axs[1].plot(time_list, v_sine_neg_list, label="Sine Negative")
axs[1].set_title("Sine Reference Signals")
axs[1].set_ylabel("Voltage (V)")
axs[1].legend()

axs[2].plot(time_list, v_phase_list)
axs[2].set_title("Phase Voltage (AC-side)")
axs[2].set_ylabel("Voltage (V)")

axs[3].plot(time_list, i_phase_list)
axs[3].set_title("Phase Current")
axs[3].set_ylabel("Current (A)")
axs[3].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
