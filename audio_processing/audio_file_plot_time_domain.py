
"""
The script loads an audio sample via librosa and then plots it in the time domain.
"""


import librosa.display
import matplotlib.pyplot as plt

# Load an example audio file provided by librosa
# The function returns the audio time series as 'audio_array' and the sampling rate
audio_array, sampling_rate = librosa.load(librosa.ex('trumpet'))

# Set up the plot with a specific width for better visibility
plt.figure(figsize=(12, 4))

# Display the waveform of the audio signal
librosa.display.waveshow(audio_array, sr=sampling_rate)

# Show the plot
plt.show()
