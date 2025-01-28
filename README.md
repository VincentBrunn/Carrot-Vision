# Carrot-Vision
# OpenCV Carrot Tracking with Kalman Filter

This project implements a carrot tracking system using OpenCV and Python. The system processes video input to track the position of a carrot in real-time, applying computer vision techniques and a Kalman Filter for smooth trajectory estimation.

## Features

- **Real-time Tracking**: Identifies and tracks a carrot in the video feed.
- **Kalman Filter Integration**: Provides predictive smoothing for the carrot's motion path.
- **Output Visualization**: Displays the carrot's position and trajectory.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- OpenCV (`cv2`)
- NumPy

Install the required libraries using pip:
```bash
pip install opencv-python-headless numpy
```

## Usage

1. Clone this repository or download the script file.
2. Run the Python script:
   ```bash
   python opencv_carrot_tracking.py
   ```
3. Provide the video input (e.g., a live webcam feed or a pre-recorded video). Adjust the source in the script if needed.

## Code Highlights

- **Detection**: The system identifies the carrot using color thresholds or contour matching.
- **Prediction**: The Kalman Filter predicts the carrot's position between frames.
- **Visualization**: Draws bounding boxes and trajectories on the video output for enhanced clarity.

## How It Works

### 1. **Input**
The system processes video input frame-by-frame, which can be sourced from a live webcam feed or a pre-recorded video file. Each frame is analyzed independently to identify and track the carrot.

### 2. **Detection**
Carrot detection is based on a combination of color filtering and contour analysis:

- **Color Filtering**: 
  - The system uses a predefined HSV (Hue, Saturation, Value) color range to isolate carrot-like colors. Typical carrot hues fall within an orange or reddish spectrum, with specific bounds defined using `cv2.inRange()`. For example:
    ```python
    lower_bound = (5, 100, 100)  # Example lower HSV bound for orange
    upper_bound = (15, 255, 255)  # Example upper HSV bound for orange
    ```
  - Pixels within this range are treated as potential carrot regions, while others are masked out.

- **Contour Analysis**:
  - After applying the mask, the system uses `cv2.findContours()` to detect distinct shapes in the filtered image.
  - It evaluates the size and shape of these contours to determine if they resemble a carrot. Criteria such as contour area and aspect ratio (long and narrow) are used for validation.
  - If multiple carrot-like contours are found, the largest or most prominent one is selected.

### 3. **Lighting Adaptation**
Lighting conditions can significantly impact detection. To adapt, the system:
- Uses HSV color space, which is less sensitive to lighting variations compared to RGB.
- Allows dynamic adjustment of the brightness (Value) threshold to accommodate different lighting environments.
- Relies on relative color values when calculating the range, making it robust to minor fluctuations in ambient light.

### 4. **Kalman Filter for Smoothing**
Once the carrot is detected, the system tracks its motion using a Kalman Filter. This predictive filter is particularly useful for:
- **Handling Occlusions**: If the carrot temporarily moves out of view or is partially blocked, the Kalman Filter predicts its position based on the motion history.
- **Reducing Noise**: By accounting for frame-to-frame variability, the Kalman Filter smooths out erratic movements caused by imperfect detection.
- **Adapting to Lighting Changes**: Even if detection momentarily fails due to sudden lighting shifts, the filter maintains continuity by predicting the carrot's location.

The Kalman Filter operates in two phases:
1. **Prediction**: Estimates the carrot's position in the next frame based on its velocity and previous positions.
2. **Correction**: Updates the prediction based on new observations from the detection step.

### 5. **Output**
The system overlays tracking annotations onto the video feed:
- A bounding box highlights the detected carrot.
- A trajectory line shows the path predicted by the Kalman Filter, offering a clear visualization of its movement.

The processed video feed is displayed in real-time, providing users with an interactive view of the carrot's motion.


## Customization

- Modify color thresholds in the `cv2.inRange()` function to adapt the detection system to different environments or objects.
- Adjust Kalman Filter parameters to suit your tracking accuracy and smoothness needs.

## Example Output

The system outputs a live video feed where the carrot's position is marked with a bounding box and its predicted trajectory is shown.

## Troubleshooting

- Ensure proper lighting for optimal detection accuracy.
- Adjust detection parameters if false positives occur or tracking fails.
