"""
Face Detection Script Using OpenCV DNN

This script performs real-time face detection from a video stream (webcam or video file).
It first checks for the required assets (a ZIP file containing model files), downloads and extracts them if necessary,
then loads a pre-trained deep learning model to detect faces in each frame captured from the video source.

Features:
- Downloads and extracts assets only if they are not already present.
- Supports webcam or video file input.
- Displays detection confidence and inference time on the video frames.

Usage:
- Run the script with an optional argument specifying the video source index or filename.
  Example: python face_detection.py 0  (for webcam)
  Example: python face_detection.py video.mp4

Dependencies:
- OpenCV (cv2)
- NumPy (optional, not used explicitly here)
"""

import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve

# Downloading Assets
def download_and_unzip(url, save_path):
    """
    Downloads a ZIP file from the specified URL and extracts its contents.

    Parameters:
    - url: URL of the ZIP file.
    - save_path: Path where the ZIP file will be saved.
    """
    print("Downloading and extracting assets...", end="")

    # Download the ZIP file
    urlretrieve(url, save_path)

    try:
        # Extract ZIP file contents in the same directory
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file or extraction error.", e)

# URL of the ZIP file containing model assets
URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

# Path where the ZIP file will be saved
asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_12.zip")

# Download the assets if they are not already present
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

#-------------

# Video source: default to webcam (index 0) or user-provided argument
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Initialize video capture object
source = cv2.VideoCapture(s)

# Create a window for display
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

#Delete the downloaded zip file
#os.remove("deploy.prototxt")
#oos.remove("res10_300x300_ssd_iter_140000_fp16.caffemodel")
os.remove("opencv_bootcamp_assets_12.zip")

while True:
    # Read a frame from the video source
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a blob from the frame for input to the neural network
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (in_width, in_height),
        mean,
        swapRB=False,
        crop=False
    )


    # Set the input to the network
    net.setInput(blob)
    # Perform forward pass to get detections
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Compute the (x, y)-coordinates of the bounding box
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            # Draw the bounding box around detected face
            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom),
                (x_right_top, y_right_top),
                (0, 255, 0),
                2
            )

            # Prepare label with confidence score
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            # Draw background rectangle for label
            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (255, 255, 255),
                cv2.FILLED
            )

            # Put label text above the bounding box
            cv2.putText(
                frame,
                label,
                (x_left_bottom, y_left_bottom),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

    # Get inference time
    t, _ = net.getPerfProfile()
    inference_time = t * 1000.0 / cv2.getTickFrequency()

    # Display inference time on frame
    label = "Inference time: %.2f ms" % inference_time
    cv2.putText(
        frame,
        label,
        (0, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # Show the frame with detections
    cv2.imshow(win_name, frame)

    # Exit loop if 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
source.release()

cv2.destroyWindow(win_name)






