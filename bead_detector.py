import cv2
import numpy as np

# Define HSV ranges for each color and their shadow variations if needed
HSV_COLOR_RANGES = {
    "red    ": [(0, 70, 50), (10, 255, 255)],
    # "red_sh ": [(160, 50, 50), (180, 255, 150)],
    "blue   ": [(90, 50, 50), (140, 255, 255)],
    "green  ": [(40, 50, 50), (85, 255, 255)],
    "yellow ": [(20, 70, 70), (35, 255, 255)],
    "black  ": [(0, 0, 0), (180, 70, 40)]
}

# def detect_color(cell):
#     """
#     Detect the dominant color in a cell using HSV color thresholds.
#     """
#     hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
#     best_color = "unknown"
#     max_pixels = 0
#     for color, (lower, upper) in HSV_COLOR_RANGES.items():
#         mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
#         pixels = cv2.countNonZero(mask)
#         if pixels > max_pixels:
#             max_pixels = pixels
#             best_color = color.strip()
#     return best_color


def detect_color(cell):
    """
    Detect the dominant color in a cell using HSV color thresholds,
    only considering the middle 50% of the cell.
    """
    height, width = cell.shape[:2]

    # Calculate the boundaries for the middle 50%
    x_start = width // 4
    x_end = width - (width // 4)
    y_start = height // 4
    y_end = height - (height // 4)

    # Extract the middle portion of the cell
    middle_portion = cell[y_start:y_end, x_start:x_end]

    # Convert to HSV
    hsv = cv2.cvtColor(middle_portion, cv2.COLOR_BGR2HSV)

    best_color = "unknown"
    max_pixels = 0

    for color, (lower, upper) in HSV_COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        pixels = cv2.countNonZero(mask)
        if pixels > max_pixels:
            max_pixels = pixels
            best_color = color.strip()

    return best_color


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


def process_grid(image_path, grid_size=(5, 5)):
    """
    Detect the colors in a grid image by:
    - Finding all colors first
    - Determining bounding box that covers all colorful objects
    - Dividing that bounding box into a grid and detecting colors
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be opened")

    # Determine bounding box that covers all colored regions
    x, y, w, h = find_bounding_box_for_colors(image)

    # Compute cell dimensions
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]
    detected_colors = [["unknown"] * grid_size[1] for _ in range(grid_size[0])]

    # For visualization
    visualized_image = image.copy()

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            y1 = y + row * cell_height
            y2 = y + (row + 1) * cell_height
            x1 = x + col * cell_width
            x2 = x + (col + 1) * cell_width

            cell = image[y1:y2, x1:x2]
            c = detect_color(cell)
            detected_colors[row][col] = c

            # Draw rectangles and label each cell
            cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visualized_image, c, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return detected_colors, visualized_image


def display_results(grid_colors, image_path):
    """
    Display the detected grid colors in a formatted way.
    """
    print(f"\nFinal Grid ({image_path}) Colors:")
    for row in grid_colors:
        print(" ".join(row))
    print("\n")


def process_camera_feed(grid_size=(5, 5)):
    """
    Process live camera feed and display the detected grid colors in real time.
    """

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:

        while True:
            # Read the current frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frane
            grid_colors, visualized_image = process_grid_frame(
                frame, grid_size)

            # Display the results
            display_results(grid_colors, frame)

            cv2.imshow("Detected Grid", visualized_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_grid_frame(frame, grid_size=(5, 5)):
    """
    Process a single frame and return the detected grid colors and a visualization.
    """
    # Determine bounding box that covers all colored regions
    x, y, w, h = find_bounding_box_for_colors(frame)

    # Compute cell dimensions
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]
    detected_colors = [["unknown"] * grid_size[1] for _ in range(grid_size[0])]

    # For visualization
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

            # Draw rectangles and label each cell
            cv2.rectangle(visualized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visualized_frame, c, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return detected_colors, visualized_frame


def main():
    try:
        process_camera_feed(grid_size=(5, 5))
    except Exception as e:
        print(f"An error occurred: {e}")

    # image_path = "img/refined_1.jpeg"  # Replace with your image path
    # grid_colors, visualized_image = process_grid(image_path, grid_size=(5, 5))
    # display_results(grid_colors, image_path)

    # cv2.imshow("Detected Grid11111", visualized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
