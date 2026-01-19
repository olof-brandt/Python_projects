import matplotlib.pyplot as plt
import numpy as np

"""
This script simulates an LC circuit driven by a sinusoidal voltage source.
It calculates the impedance of the inductor and capacitor at a given frequency,
computes the current flowing through the circuit, and plots both the source voltage
and the imaginary part of the current over time.

Key Steps:
- Define component values (Capacitor C, Inductor L) and the excitation frequency.
- Calculate angular frequency and impedances.
- Generate a time array for simulation.
- Calculate the source voltage as a cosine wave.
- Compute the circuit current based on total impedance.
- Plot the source voltage and the imaginary part of the current.
"""

# Component values
C = 620e-9  # Capacitance in Farads (F)
L = 100e-6  # Inductance in Henrys (H)
f = 20e3    # Frequency in Hertz (Hz)

# Calculate angular frequency
w = 2 * np.pi * f

# Impedances of the inductor and capacitor
ZL = 1j * w * L
ZC = 1 / (1j * w * C)

# Time settings for simulation
dt = 1 / (40 * f)  # Time step to ensure smooth plots
t_start = 0
t_stop = 2 / f  # Two periods of the wave

# Generate time array
t = np.arange(t_start, t_stop, dt)

# Voltage source: sinusoidal voltage
V = 24 * np.cos(2 * np.pi * f * t)

# Total impedance of the circuit (assuming R=0 for simplicity)
R = 0
Z_total = R + ZL + ZC

# Calculate the current in the circuit
I = V / Z_total

# Plotting the voltage and the imaginary part of the current
plt.figure(figsize=(10, 6))
plt.plot(t, V, label='Voltage $V$ (V)')
plt.plot(t, np.imag(I), label='Imaginary part of current $Im(I)$ (A)')

# Add title, axis labels, legend, and grid
plt.title('Voltage and Current in LC Circuit at Resonance')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
