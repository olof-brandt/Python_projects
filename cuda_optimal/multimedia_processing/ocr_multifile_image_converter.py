# This script performs Optical Character Recognition (OCR) on a collection of images stored in a specified directory.
# It leverages the Tesseract OCR engine via the pytesseract Python wrapper to extract text from images.
# The process involves installing necessary dependencies, mounting Google Drive to access images,
# locating image files with common formats, performing OCR on each image, and saving the extracted text into an output file.
# This example is configured to work with images containing Swedish text ('swe').

!pip install pytesseract pillow
!sudo apt-get install tesseract-ocr-swe



import os
import glob
import pytesseract
from PIL import Image

# Optional: specify the path to the tesseract executable if needed
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def find_image_files(directory):
    # Common image extensions
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

def perform_ocr_on_images(image_files, output_txt):
    with open(output_txt, 'w', encoding='utf-8') as outfile:
        for image_path in sorted(image_files):
            try:
                print(f"Processing {image_path}...")
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang='swe')
                #outfile.write(f"\n--- Text from {os.path.basename(image_path)} ---\n")
                outfile.write(text)
                outfile.write("\n")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def main():
    directory = 'drive/MyDrive/input_folder'
    output_file = directory + '/output.txt'
    #output_file = input("Enter the desired output text file name (e.g., output.txt): ").strip()

    if not os.path.isdir(directory):
        print("The specified directory does not exist.")
        return

    image_files = find_image_files(directory)

    if not image_files:
        print("No image files found in the specified directory.")
        return

    perform_ocr_on_images(image_files, output_file)
    print(f"OCR complete. Text saved to {output_file}.")

if __name__ == "__main__":
    main()
