import cv2
import numpy as np

# Define HSV ranges for each color
HSV_COLOR_RANGES = {
    "red": [ (0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255) ],
    "purple": [ (130, 50, 50), (160, 255, 255) ],
    "blue": [ (100, 50, 50), (140, 255, 255) ],
    "green": [ (40, 50, 50), (85, 255, 255) ],
    "yellow": [ (20, 70, 70), (35, 255, 255) ],
    "black": [ (0, 0, 0), (180, 20, 40) ]
}

def detect_color(cell):
    # Apply Gaussian blur to reduce noise
    cell = cv2.GaussianBlur(cell, (5,5), 0)
    
    # Convert to HSV
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    
    # Initialize a mask to keep track of assigned pixels
    mask_assigned = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
    
    # Color priority list
    color_priority = [
        ('red', HSV_COLOR_RANGES["red"]),
        ('purple', HSV_COLOR_RANGES["purple"]),
        ('blue', HSV_COLOR_RANGES["blue"]),
        ('green', HSV_COLOR_RANGES["green"]),
        ('yellow', HSV_COLOR_RANGES["yellow"]),
        ('black', HSV_COLOR_RANGES["black"])
    ]
    
    # Dictionary to hold pixel counts for each color
    color_counts = {color: 0 for color, _ in color_priority}
    
    for color, ranges in color_priority:
        if len(ranges) == 4:
            # Red has two ranges
            lower1 = np.array(ranges[0])
            upper1 = np.array(ranges[1])
            lower2 = np.array(ranges[2])
            upper2 = np.array(ranges[3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(ranges[0])
            upper = np.array(ranges[1])
            mask = cv2.inRange(hsv, lower, upper)
        
        # Exclude pixels already assigned
        mask_unassigned = cv2.bitwise_and(mask, cv2.bitwise_not(mask_assigned))
        
        # Count pixels for this color
        color_counts[color] += cv2.countNonZero(mask_unassigned)
        
        # Assign these pixels to avoid double-counting
        mask_assigned = cv2.bitwise_or(mask_assigned, mask_unassigned)
    
    # Determine the dominant color
    dominant_color = max(color_counts, key=color_counts.get)
    
    return dominant_color

def process_grid(image_path, grid_size=(5, 5)):
    """
    Detect the colors in a grid image using a hard-coded grid position.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be opened")

    # Hard-coded grid position and size based on pixel values
    x = 500  # Starting x-coordinate of the grid
    y = 150  # Starting y-coordinate of the grid
    w = 800  # Width of the grid
    h = 650  # Height of the grid

    # Ensure the grid region is within the image boundaries
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("Grid coordinates are out of image bounds")

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

def main():
    image_path = "noice1.jpg"  # Replace with your image path
    grid_colors, visualized_image = process_grid(image_path, grid_size=(5, 5))
    display_results(grid_colors, image_path)
    
    cv2.imshow("Detected Grid", visualized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()