"""
This script uses the Hugging Face transformers library to convert a given text into speech.
It utilizes the "suno/bark-small" model for text-to-speech synthesis.
The resulting audio is then played within an IPython environment (such as Jupyter Notebook).
"""

from transformers import pipeline
from IPython.display import Audio

# Initialize the text-to-speech pipeline with the specified model
tts_pipeline = pipeline("text-to-speech", model="suno/bark-small")

# Define the text to be converted to speech
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy."

# Generate the speech audio from the text
output = tts_pipeline(text)

# Play the generated audio within an IPython environment
Audio(output["audio"], rate=output["sampling_rate"])
