import cv2
import numpy as np
import time
from datetime import datetime

# Your existing HSV_COLOR_RANGES dictionary remains the same
HSV_COLOR_RANGES = {
    "red    ": [(0, 70, 50), (10, 255, 255)],
    "blue   ": [(90, 50, 50), (140, 255, 255)],
    "green  ": [(40, 50, 50), (85, 255, 255)],
    "yellow ": [(20, 70, 70), (35, 255, 255)],
    "black  ": [(0, 0, 0), (180, 70, 40)]
}

def find_bounding_box_for_colors(image):
    """
    Create a combined mask of all colors, then find the minimal bounding rectangle 
    covering all detected color pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Combine all color masks
    combined_mask = None
    for color, (lower, upper) in HSV_COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # If no colors are found, default to full image
    if combined_mask is None or cv2.countNonZero(combined_mask) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    # Find all non-zero points in combined_mask
    pts = cv2.findNonZero(combined_mask)
    if pts is None:
        # No colored objects found
        return 0, 0, image.shape[1], image.shape[0]
    # Get bounding rectangle of all colored points
    x, y, w, h = cv2.boundingRect(pts)
    return x, y, w, h

def process_frame(frame, grid_size=(5, 5)):
    """
    Process a single frame from the video feed
    """
    x, y, w, h = find_bounding_box_for_colors(frame)
    
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]
    detected_colors = [["unknown"] * grid_size[1] for _ in range(grid_size[0])]
    
    visualized_frame = frame.copy()

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            y1 = y + row * cell_height
            y2 = y + (row + 1) * cell_height
            x1 = x + col * cell_width
            x2 = x + (col + 1) * cell_width
            
            cell = frame[y1:y2, x1:x2]
            c = detect_color(cell)
            detected_colors[row][col] = c
            
            cv2.rectangle(visualized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visualized_frame, c, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return detected_colors, visualized_frame

def save_to_file(grid_colors, filename="color_detections.txt"):
    """
    Append detection results to a file with timestamp
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"\nDetection Time: {timestamp}\n")
        for row in grid_colors:
            f.write(" ".join(row) + "\n")
        f.write("-" * 50 + "\n")

def main(camera_index=0, detection_interval=1.0):
    """
    Main function to process live video feed
    camera_index: index of the camera to use (usually 0 for built-in webcam)
    detection_interval: time in seconds between detections
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    last_detection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        current_time = time.time()
        
        # Perform detection at specified interval
        if current_time - last_detection_time >= detection_interval:
            grid_colors, visualized_frame = process_frame(frame)
            save_to_file(grid_colors)
            last_detection_time = current_time
            
            # Display the frame
            cv2.imshow("Live Color Detection", visualized_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start detection with camera index 0 and 1-second interval
    main(camera_index=0, detection_interval=1.0)
