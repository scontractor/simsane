import cv2
import numpy as np
from tqdm import tqdm

# Load the image
image_path = 'motor_screenshot.jpg'
print("Loading the image...")
image = cv2.imread(image_path)

# Convert the image to HSV color space
print("Converting to HSV...")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color ranges for segmentation (adjust these as needed)
color_ranges = {
    'pink': ((140, 100, 100), (170, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'red': ((0, 100, 100), (10, 255, 255)),
    'green': ((40, 100, 100), (70, 255, 255))
}

# Minimum contour area to consider (adjust to filter out small noise)
min_contour_area = 100  # You can adjust this value

# Initialize a list to hold all detected contours and their dimensions
all_contours = []
all_dimensions = []

# Loop over each color range to detect objects of that color
for color_name, (lower, upper) in color_ranges.items():
    print(f"Processing color: {color_name}")

    # Create a mask for the current color range
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours for the current color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours and store the rest
    for contour in tqdm(contours, desc=f"Calculating dimensions for {color_name}"):
        if cv2.contourArea(contour) >= min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            all_contours.append(contour)
            all_dimensions.append((x, y, w, h))

# Draw contours, bounding boxes, and dimension annotations on the original image
print("Drawing bounding boxes and dimensions...")
for i, (contour, (x, y, w, h)) in enumerate(zip(all_contours, all_dimensions)):
    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Annotate with dimensions, adjust text position and font size for readability
    cv2.putText(image, f"W: {w}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, f"H: {h}px", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Save the image with annotations
output_path = '/mnt/data/motor_screenshot_cleaned_dimensions.jpg'
cv2.imwrite(output_path, image)
print(f"Annotated image saved as {output_path}")

# Display the image with contours and bounding boxes
print("Displaying the image...")
cv2.imshow('Detected Contours with Dimensions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the dimensions
print("Printing the dimensions of detected contours...")
for i, (x, y, w, h) in enumerate(all_dimensions):
    print(f"Contour {i + 1}: x={x}, y={y}, width={w}px, height={h}px")

print("Processing complete.")
