"""
This script loads an audio sample from librosa's example dataset, performs various spectral analyses,
and visualizes the results. It demonstrates how to compute and plot a short-time Fourier transform (STFT),
a Mel spectrogram, and a raw Fourier spectrum.

Dependencies:
- librosa: for audio processing
- matplotlib: for plotting
- numpy: for numerical operations
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an example audio file (such as a trumpet sound)
audio, sampling_rate = librosa.load(librosa.ex("trumpet"))

# Optional: Plot the waveform (commented out)
# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(audio, sr=sampling_rate)
# plt.title("Waveform of the audio sample")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

# Select a segment of the audio for Fourier analysis
dft_input = audio[:4096]

# Calculate the windowed Fourier Transform (DFT) of the segment
window = np.hanning(len(dft_input))  # Apply a Hann window to reduce spectral leakage
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)  # Compute the real FFT

# Convert the amplitude spectrum to decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# Get the corresponding frequency bins
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))


"""
# Optional: Plot the raw Fourier spectrum in decibels versus frequency
plt.figure(figsize=(12, 4))
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.title("Fourier Spectrum of a Segment")
plt.xscale("log")  # Use logarithmic scale for better visualization
plt.show()
"""

# --- Alternative spectral analysis methods (commented out) ---

# Short-Time Fourier Transform (STFT) and Spectrogram
"""
D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar(label="Amplitude (dB)")
plt.title("STFT Spectrogram")
plt.show()
"""

# Mel Spectrogram
S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot the Mel spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar(label="Amplitude (dB)")
plt.title("Mel Spectrogram")
plt.show()
