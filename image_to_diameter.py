import cv2
import numpy as np
import csv

def measure_diameters(image_path, stator_diameter_mm):
    """
    Measures the diameters of circular objects in an image based on the known stator diameter.

    Args:
        image_path: Path to the image file.
        stator_diameter_mm: Known diameter of the stator in millimeters.

    Returns:
        A list of measured diameters in millimeters.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 100, 200)

    # Detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        diameters_px = [c[2] * 2 for c in circles]  # Calculate diameters in pixels

        # Assuming pixel-to-mm ratio is proportional to stator diameter
        pixel_to_mm_ratio = stator_diameter_mm / max(diameters_px)
        diameters_mm = [d * pixel_to_mm_ratio for d in diameters_px]

        return diameters_mm

    else:
        return []

# Example usage
image_file = r"C:\Users\scontractor\PycharmProjects\simsane\simsane\motor_screenshot.jpg"
stator_diameter = 50  # mm
diameters = measure_diameters(image_file, stator_diameter)

# Write diameters to CSV file
with open('diameters.csv', 'w', newline='') as csvfile:
    fieldnames = ['diameter_mm']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for diameter in diameters:
        writer.writerow({'diameter_mm': diameter})
