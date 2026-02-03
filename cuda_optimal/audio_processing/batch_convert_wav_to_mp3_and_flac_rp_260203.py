


# Install necessary packages
!pip install pydub
!ap t -get install ffmpeg -y

from pydub import AudioSegment
import os

!apt-get update
!apt-get install -y ffmpeg
!ffmpeg -version

##################



"""
# This script processes audio files in a specified directory by converting each WAV file to both FLAC and MP3 formats, 
saving the converted files into an output directory. 
Additionally, it collects all audio segments into a list for potential further concatenation or processing.
"""

# Specify your directory in Google Drive
# Replace 'your_directory_path' with the path relative to your drive
input_dir = "audio_processing"

# Output directories
output_dir = input_dir + "/converted_audio"
os.makedirs(output_dir, exist_ok=True)

# Initialize list to hold all audio segments for final MP3
all_audio_segments = []

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.wav'):
        wav_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # Convert WAV to FLAC
        flac_path = os.path.join(output_dir, base_name + ".flac")
        audio = AudioSegment.from_wav(wav_path)
        audio.export(flac_path, format="flac")
        print(f"Converted {wav_path} to {output_dir}")

        # combined_audio = sum(all_audio_segments)
        mp3_output_path = os.path.join(output_dir, base_name + ".mp3")
        audio.export(mp3_output_path, format="mp3", bitrate="320k")
        print(f"Converted MP3 saved at {output_dir}")

        # Append to list for concatenation
        all_audio_segments.append(audio)
























