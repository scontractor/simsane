import cv2
import numpy as np
import csv
from tqdm import tqdm

# Load the image
image_path = 'motor_screenshot.jpg'
print("Loading the image...")
image = cv2.imread(image_path)

# Convert the image to HSV color space
print("Converting to HSV...")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for the pink area (adjust as needed)
pink_lower = np.array([140, 100, 100])
pink_upper = np.array([170, 255, 255])

# Create a mask for the pink area
pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

# Find contours in the pink area
pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest pink contour is the one we're interested in
pink_contour = max(pink_contours, key=cv2.contourArea)

# Get the bounding box of the pink contour
x, y, w, h = cv2.boundingRect(pink_contour)

# Take user input for the known diameter of the pink area
known_diameter_mm = float(input("Enter the known diameter of the pink area in millimeters: "))

# Calculate the scale (pixels per millimeter)
diameter_pixels = max(w, h)
scale = known_diameter_mm / diameter_pixels
print(f"Scale calculated: {scale:.4f} pixels per millimeter")

# Define other color ranges (adjust as needed)
color_ranges = {
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'red': ((0, 100, 100), (10, 255, 255)),
    'green': ((40, 100, 100), (70, 255, 255))
}

# Minimum contour area to consider (adjust to filter out small noise)
min_contour_area = 100  # Adjust as necessary

# Initialize lists to hold all detected contours and their dimensions
all_contours = []
all_dimensions_mm = []

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
            w_mm = w * scale
            h_mm = h * scale
            all_contours.append(contour)
            all_dimensions_mm.append((x, y, w_mm, h_mm))

# Draw contours, bounding boxes, and dimension annotations on the original image
print("Drawing bounding boxes and dimensions...")
for i, (contour, (x, y, w_mm, h_mm)) in enumerate(zip(all_contours, all_dimensions_mm)):
    # Draw the bounding box in green
    cv2.rectangle(image, (x, y), (x + int(w_mm / scale), y + int(h_mm / scale)), (0, 255, 0), 2)
    # Annotate with dimensions in mm
    cv2.putText(image, f"W: {w_mm:.2f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(image, f"H: {h_mm:.2f}mm", (x, y + int(h_mm / scale) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                1)

# Save the image with annotations
output_image_path = '/mnt/data/motor_screenshot_with_mm_dimensions.jpg'
cv2.imwrite(output_image_path, image)
print(f"Annotated image saved as {output_image_path}")

# Save the dimensions to a CSV file
output_csv_path = 'dimensions_in_mm.csv'
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Contour Number", "X (pixels)", "Y (pixels)", "Width (mm)", "Height (mm)"])
    for i, (x, y, w_mm, h_mm) in enumerate(all_dimensions_mm):
        writer.writerow([i + 1, x, y, f"{w_mm:.2f}", f"{h_mm:.2f}"])

print(f"Dimensions saved to {output_csv_path}")

# Display the image with contours and bounding boxes
print("Displaying the image...")

import matplotlib.pyplot as plt

# After saving the image:
output_image_path = 'motor_screenshot_with_mm_dimensions.jpg'
cv2.imwrite(output_image_path, image)

# Optionally display the image using matplotlib
print("Displaying the image using matplotlib...")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()

print("Processing complete.")
