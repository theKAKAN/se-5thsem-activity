import os
import cv2

def process_image(input_path, output_path):
    # Load the image using OpenCV
    image = cv2.imread(input_path)
    # Apply Gaussian blur using OpenCV
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply Median blur to the above image
    blurred_image = cv2.medianBlur( blurred_image, 5 )

    # Before saving, check if the directory exists
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)

def process_images_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                # Find the relative path to preserve the directory
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
                process_image(input_path, output_path)

input_folder = 'Images'
output_folder = 'Output_q1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder(input_folder, output_folder)
