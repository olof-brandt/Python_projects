"""
This script uses the Hugging Face Diffusers library to generate an image from a text prompt.
It loads a pre-trained image generation pipeline, generates an image based on the given prompt,
and saves the resulting image to a file.

"""

from diffusers import AutoPipelineForText2Image
import torch

# Load the pre-trained pipeline with float16 precision for faster inference
# and move it to the GPU for acceleration.
pipeline = AutoPipelineForText2Image.from_pretrained(
    'dataautogpt3/OpenDalleV1.1',  # Path or model identifier
    torch_dtype=torch.float16
).to('cuda')

# Define the text prompt for image generation
prompt = 'flying cars in Stockholm'

# Generate the image based on the prompt
generated_images = pipeline(prompt).images
image = generated_images[0]  # Get the first generated image

# Specify the filename for saving the image
filename = "flying_cars_stockholm.jpg"

# Save the generated image to disk
image.save(filename)

print(f"Image saved as {filename}")
