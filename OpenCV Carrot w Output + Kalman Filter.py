import cv2
import numpy as np
from collections import deque

# Adjusted HSV range for orange
orange_lower = np.array([5, 180, 160])
orange_upper = np.array([25, 255, 255])

# Green color range remains the same
green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])

# Store previous bounding box positions for the past 15 frames
orange_rect_history = deque(maxlen=15)  # Store up to 15 frames worth of bounding box positions

# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic params (x, y, dx, dy), 2 measured params (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]], np.float32) * 0.03

def track_carrot(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for orange and green with the adjusted HSV ranges
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # Apply stronger morphological operations to smooth the detection
    kernel = np.ones((9, 9), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

    # Find contours for both masks
    contours_orange, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Track the carrot if contours are found
    if contours_orange and contours_green:
        # Filter by area and get the largest contour for orange and green
        min_contour_area = 1200
        largest_orange_contour = max(contours_orange, key=cv2.contourArea)
        largest_green_contour = max(contours_green, key=cv2.contourArea)

        if cv2.contourArea(largest_orange_contour) > min_contour_area:
            # Get the bounding box of the largest orange contour
            orange_rect = cv2.minAreaRect(largest_orange_contour)
            green_rect = cv2.minAreaRect(largest_green_contour)

            # Store the center of the orange box in the history buffer
            orange_rect_history.append(orange_rect[0])

            # Calculate the average center position
            if len(orange_rect_history) > 1:
                avg_center = np.mean(orange_rect_history, axis=0)
            else:
                avg_center = orange_rect[0]

            # Kalman filter prediction
            predicted = kalman.predict()

            # Kalman filter correction based on the measurement
            measured = np.array([[np.float32(avg_center[0])], [np.float32(avg_center[1])]])
            kalman.correct(measured)

            # Use the Kalman filter's predicted position to stabilize tracking
            kalman_x, kalman_y = predicted[0], predicted[1]

            # Draw the corrected bounding box using Kalman prediction
            orange_box = cv2.boxPoints(orange_rect)
            orange_box[:, 0] += (kalman_x - orange_rect[0][0])  # Apply Kalman x adjustment
            orange_box[:, 1] += (kalman_y - orange_rect[0][1])  # Apply Kalman y adjustment
            orange_box = np.int64(orange_box)

            green_box = cv2.boxPoints(green_rect)
            green_box = np.int64(green_box)

            # Draw the adjusted contours on the frame
            cv2.drawContours(frame, [orange_box], 0, (0, 165, 255), 2)
            cv2.drawContours(frame, [green_box], 0, (0, 255, 0), 2)

            # Get the x-coordinate of the Kalman-predicted carrot's center
            carrot_x = int(kalman_x)
            print(f"Kalman-Predicted Carrot X center: {carrot_x}")  # Output the x value

            # Check if the two bounding boxes are adjacent
            distance = np.linalg.norm(np.array(orange_rect[0]) - np.array(green_rect[0]))
            if distance < 50000:
                cv2.putText(frame, "Carrot detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

def main():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to track the carrot
        result_frame = track_carrot(frame)

        # Display the result
        cv2.imshow("Carrot Tracker with Kalman Filter", result_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
