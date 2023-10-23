import os
import cv2
from PIL import Image

def process_image(input_path, output_path):
    # Load the image using OpenCV
    # Read as grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Convert it into a binary image
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    # Resize
    resized_image = Image.fromarray(binary_image)
    resized_image = resized_image.resize((64, 64))

    # Before saving, check if the directory exists
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save the blurred image
    resized_image.save(output_path)

def process_images_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                # Find the relative path to preserve the directory
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
                process_image(input_path, output_path)

input_folder = 'Output_q1'
output_folder = 'Output_q2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder(input_folder, output_folder)
