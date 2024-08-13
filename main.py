import cv2


def calculate_dimensions(image_path, known_dimension, dimension_type='width'):
    # Load the image
    image = cv2.imread(image_path)

    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    if dimension_type == 'width':
        # Calculate the height based on the known width
        ratio = known_dimension / original_width
        new_width = known_dimension
        new_height = int(original_height * ratio)
    elif dimension_type == 'height':
        # Calculate the width based on the known height
        ratio = known_dimension / original_height
        new_height = known_dimension
        new_width = int(original_width * ratio)
    else:
        raise ValueError("dimension_type must be 'width' or 'height'")

    return new_width, new_height


# Get image path from the user
image_path = input("Enter the path to your image: ")

# Get known dimension from the user
known_dimension = int(input("Enter the known dimension (width/height): "))

# Get the dimension type from the user (width or height)
dimension_type = input("Is the known dimension width or height? (Enter 'width' or 'height'): ").lower()

# Calculate the new dimensions
new_width, new_height = calculate_dimensions(image_path, known_dimension, dimension_type)

# Output the new dimensions
print(f"New dimensions (Width x Height): {new_width} x {new_height}")
