import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Load the image
image_path = 'motor_screenshot.jpg'
image = cv2.imread(image_path)

# Get image dimensions
height, width = image.shape[:2]

# Calculate the geometric center of the image
origin = np.array([width / 2, height / 2], dtype=np.float64)

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for potting (yellow) and winding (orange)
color_ranges = {
    'potting': ((20, 100, 100), (30, 255, 255)),  # Yellow color range
    'winding': ((10, 100, 100), (20, 255, 255))  # Orange color range
}

# Initialize a dictionary to hold distances for each color
distances_mm = {}

# Take user input for the known diameter of the pink area
known_diameter_mm = float(input("Enter the known diameter of the pink area in millimeters: "))

# Create a mask for the pink area to calculate scale
pink_lower = np.array([140, 100, 100])
pink_upper = np.array([170, 255, 255])
pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pink_contour = max(pink_contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(pink_contour)
diameter_pixels = max(w, h)
scale = known_diameter_mm / diameter_pixels

# Draw the origin on the image
cv2.circle(image, (int(origin[0]), int(origin[1])), 5, (255, 0, 0), -1)

# Loop over each color range to detect the largest contour of that color
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the target
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the closest point on the contour to the origin
        min_distance = float('inf')
        closest_point = None
        for point in largest_contour:
            point = point[0]
            distance_to_origin = np.linalg.norm(point - origin)
            if distance_to_origin < min_distance:
                min_distance = distance_to_origin
                closest_point = point

        if closest_point is not None:
            # Calculate the distance in mm
            closest_distance_mm = min_distance * scale
            distances_mm[color_name] = closest_distance_mm

            # Draw the closest point
            cv2.circle(image, tuple(closest_point), 5, (0, 0, 255), -1)
            cv2.putText(image, f"{color_name.capitalize()} D: {closest_distance_mm:.2f}mm", (closest_point[0] - 40, closest_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw line from origin to the closest point
            cv2.line(image, (int(origin[0]), int(origin[1])), tuple(closest_point), (0, 0, 255), 2)

# Save the image with annotations
output_image_path = 'motor_screenshot_with_mm_dimensions.jpg'
cv2.imwrite(output_image_path, image)

# Save the distances to a CSV file with the new name
output_csv_path = 'detected_radii_in_mm.csv'
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Color", "Distance from Origin (mm)"])
    for color_name, distance_mm in distances_mm.items():
        writer.writerow([color_name.capitalize(), f"{distance_mm:.2f}"])

# Display the image with annotations
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()