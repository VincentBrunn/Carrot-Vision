# Carrot-Vision
Demo OpenCV Project that uses local or external camera to identify any carrots in frame.

---

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

1. **Input**: The video feed is processed frame-by-frame.
2. **Detection**: Identifies carrot-like objects based on predefined criteria.
3. **Filtering**: Uses a Kalman Filter to estimate and smooth the motion of the detected carrot.
4. **Output**: The processed video with tracking annotations is displayed.

## Customization

- Modify color thresholds in the `cv2.inRange()` function to adapt the detection system to different environments or objects.
- Adjust Kalman Filter parameters to suit your tracking accuracy and smoothness needs.

## Example Output

The system outputs a live video feed where the carrot's position is marked with a bounding box and its predicted trajectory is shown.

## Troubleshooting

- Ensure proper lighting for optimal detection accuracy.
- Adjust detection parameters if false positives occur or tracking fails.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
