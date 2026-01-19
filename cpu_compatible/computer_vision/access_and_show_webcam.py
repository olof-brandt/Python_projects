"""
Effortless Live Video Streaming with OpenCV:
This script captures live video from a specified source (webcam or video file)
and displays it in a resizable window. You can specify the video source via
a command-line argument; if none is provided, it defaults to the primary webcam.

Usage:
    python script_name.py [video_source]

    - video_source can be an integer index (e.g., 0, 1, 2, ...) for webcams,
      or a filename for a video file.

Example:
    python script_name.py 0            # Use default webcam
    python script_name.py video.mp4  # Play video file
"""

import cv2
import sys

# Welcome message
print("Starting video capture...")

# Default source is 0 (primary webcam)
source_index = 0

# If a command-line argument is provided, use it as the video source
if len(sys.argv) > 1:
    source_arg = sys.argv[1]
    # If the argument is a digit, convert to integer for device index
    if source_arg.isdigit():
        source_index = int(source_arg)
    else:
        source_index = source_arg

# Initialize video capture with the specified source
cap = cv2.VideoCapture(source_index)

# Create a window for display with a resizable property
window_name = 'Camera Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Main loop to read frames until 'Esc' key is pressed
while True:
    has_frame, frame = cap.read()
    if not has_frame:
        print("No more frames or cannot access source.")
        break

    # Display the current frame in the window
    cv2.imshow(window_name, frame)

    # Wait for 1 ms and check if 'Esc' (27) is pressed to exit
    if cv2.waitKey(1) == 27:
        print("Escape key pressed. Exiting.")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyWindow(window_name)
