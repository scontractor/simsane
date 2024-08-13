import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Load the image
image_path = 'motor_screenshot.jpg'
image = cv2.imread(image_path)

# Convert the image to HSV color space
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

# Define other color ranges (adjust as needed)
color_ranges = {
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'red': ((0, 100, 100), (10, 255, 255)),
    'green': ((40, 100, 100), (70, 255, 255))
}

# Initialize lists to hold all detected circles and their distances
all_circles = []
all_distances_mm = []

# Origin point from which to measure the distance
origin = np.array([255, 0], dtype=np.float64)

# Loop over each color range to detect circles of that color
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, lower, upper)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = np.array([i[0], i[1]], dtype=np.float64)
            radius = i[2]
            distance_pixels = np.linalg.norm(center - origin)
            distance_mm = distance_pixels * scale
            all_circles.append((center, radius))
            all_distances_mm.append(distance_mm)

            cv2.circle(image, tuple(center.astype(int)), radius, (0, 255, 0), 2)
            cv2.circle(image, tuple(center.astype(int)), 2, (0, 0, 255), 3)
            cv2.putText(image, f"D: {distance_mm:.2f}mm", (int(center[0]) - 40, int(center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Save the image with annotations
output_image_path = 'motor_screenshot_with_mm_dimensions.jpg'
cv2.imwrite(output_image_path, image)

# Save the distances to a CSV file
output_csv_path = 'distances_in_mm.csv'
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Circle Number", "Center X (pixels)", "Center Y (pixels)", "Distance from Origin (mm)"])
    for i, ((center, radius), distance_mm) in enumerate(zip(all_circles, all_distances_mm)):
        writer.writerow([i + 1, int(center[0]), int(center[1]), f"{distance_mm:.2f}"])

# Display the image with circles and distance annotations
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()