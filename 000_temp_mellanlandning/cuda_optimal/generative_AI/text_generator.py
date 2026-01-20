# Import the necessary library from the transformers package
from transformers import pipeline

# Initialize a text-generation pipeline with the 'distilgpt2' model
# This pipeline will generate text based on a given prompt
generator = pipeline("text-generation", model="distilgpt2")

# Generate text based on the prompt "Warren Buffet used to"
# - max_length: maximum length of the generated sequence
# - num_return_sequences: number of different sequences to generate
res = generator(
    "Warren Buffet used to",
    max_length=100,
    num_return_sequences=2
)

# Print the generated texts
print(res)
