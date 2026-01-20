!pip install faster_whisper

import os
from pydub import AudioSegment
import zipfile
import numpy as np
import librosa
from faster_whisper import WhisperModel

# Configuration
root_directory = 'drive/MyDrive'#'content/drive/MyDrive'
data_dir = root_directory + '/soundfile_splitter_transcriber' #+ '/20260103_vildsvinsdetektor'

INPUT_FOLDER = data_dir + '/input'
OUTPUT_FOLDER = data_dir + '/output'
FILENAME = 'teknik båt 20251202 reparation och service'  # Replace with your actual filename
MP3_FILENAME = FILENAME + '.mp3'
ZIP_FILENAME = 'split_' + FILENAME + '.zip'

CHUNK_DURATION_MINUTES = 4
CHUNK_DURATION_SECONDS = 45
OVERLAP_SECONDS = 15

def main():
    # Construct file paths
    input_path = os.path.join(INPUT_FOLDER, MP3_FILENAME)
    output_zip_path = os.path.join(OUTPUT_FOLDER, ZIP_FILENAME)

    # Make sure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load audio with librosa to get raw audio data
    audio, sr = librosa.load(input_path, sr=None)  # sr=None preserves original sample rate

    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)

    # Create an AudioSegment from raw audio data
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit audio
        channels=1
    )

    # Initialize the WhisperModel once
    #model_size = "base"  # or "small", "medium", "large", depending on your needs
    #model = WhisperModel(model_size, device="cpu")  # or "cuda" if GPU is available


    model_size = "large-v2"

# Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Convert durations to milliseconds
    chunk_duration = (CHUNK_DURATION_MINUTES * 60 + CHUNK_DURATION_SECONDS) * 1000
    overlap = OVERLAP_SECONDS * 1000

    chunks = []

    start_time = 0
    audio_length = len(audio_segment)

    print(f"Total audio length (ms): {audio_length}")

    # Generate overlapping chunks
    while start_time < audio_length:
        end_time = start_time + chunk_duration
        if end_time > audio_length:
            end_time = audio_length

        chunk = audio_segment[start_time:end_time]
        chunks.append(chunk)

        start_time += (chunk_duration - overlap)
        if start_time >= audio_length:
            break

    # Prepare ZIP archive
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for i, chunk in enumerate(chunks):
            filename = f'chunk_{i+1}.mp3'
            chunk.export(filename, format='mp3')

            # Load the exported chunk as raw audio for whisper
            # Convert to numpy array
            y, sr_chunk = librosa.load(filename, sr=None)
            # Transcribe using faster_whisper

            """
            segments = model.transcribe(y, beam_size=5, best_of=3)


            # Combine all transcriptions
            transcription = ' '.join(segment.text for segment in segments)


            segments = list(model.transcribe(y, beam_size=5, best_of=3))
            #transcription = ' '.join(segment.text for segment in segments)

            # Print or store the transcription
            print(segments.word)
            #print(f"Transcription for chunk {i+1}:\n{transcription}\n")
            """
            #from faster_whisper import WhisperModel


            segments, info = model.transcribe(filename, beam_size=5, language='sv')

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            previous_text = ""

            for segment in segments:
                if segment.text != previous_text:
                    print("%s" % (segment.text))
                    previous_text = segment.text

            # Optionally, you can write transcriptions to a file or embed in zip as metadata
            # For now, just add the audio chunk to the zip
            zipf.write(filename, arcname=filename)
            os.remove(filename)

    print(f"Created ZIP archive with {len(chunks)} chunks at: {output_zip_path}")

if __name__ == "__main__":
    main()
