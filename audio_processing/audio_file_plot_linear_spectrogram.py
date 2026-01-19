"""
This script loads an audio sample, performs spectral analysis, and visualizes the results.
It demonstrates how to:
- Load an audio file with librosa
- Compute and plot a segment's Fourier Transform
- Compute and plot the spectrogram of the entire audio

Ensure you have librosa, numpy, and matplotlib installed:
pip install librosa matplotlib numpy
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an example audio sample from librosa's included datasets
audio, sampling_rate = librosa.load(librosa.ex("trumpet"))

# Optional: Plot the waveform (uncomment if needed)
# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(audio, sr=sampling_rate)
# plt.title("Waveform of the audio sample")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

# Select a segment of the audio for Fourier analysis (first 4096 samples)
segment = audio[:4096]

# Perform a windowed Fourier Transform on the selected segment
window = np.hanning(len(segment))
windowed_segment = segment * window

# Calculate the real-valued FFT
dft = np.fft.rfft(windowed_segment)

# Obtain the amplitude spectrum in decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# Generate corresponding frequency bins
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(segment))


"""
# Optional: Plot the amplitude spectrum of the segment
plt.figure(figsize=(12, 4))
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.title("Fourier Spectrum of a Segment")
plt.xscale("log")
plt.grid(True)
plt.show()
"""

# Compute the Short-Time Fourier Transform (STFT) of the entire audio
D = librosa.stft(audio)

# Convert the amplitude of STFT to decibel scale
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar(label="Amplitude (dB)")
plt.title("Spectrogram of the Audio Sample")
plt.show()
