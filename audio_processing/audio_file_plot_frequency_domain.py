"""
This script loads an audio sample using librosa, computes its Short-Time Fourier Transform (STFT),
and plots the amplitude spectrum in decibels over frequency on a logarithmic scale.
It demonstrates basic digital signal processing steps: loading audio, windowing, FFT computation,
and visualization of the frequency domain representation.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an example audio sample (trumpet sound) with librosa
audio, sampling_rate = librosa.load(librosa.ex('trumpet'))

# Select a segment of the audio for analysis (first 4096 samples)
dft_input = audio[:4096]

# Apply a Hanning window to reduce spectral leakage
window = np.hanning(len(dft_input))
windowed_input = dft_input * window

# Compute the real FFT of the windowed segment
dft = np.fft.rfft(windowed_input)

# Calculate the amplitude spectrum in decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# Generate frequency bins corresponding to the FFT result
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

# Plot the amplitude spectrum in decibels versus frequency
plt.figure(figsize=(12, 4))
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.title("Amplitude Spectrum of Audio Segment")
plt.xscale("log")  # Logarithmic frequency scale
plt.tight_layout()
plt.show()
