import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

def process_image(input_path, output_path):
    # Load the image using OpenCV
    # Read as grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Skeletonize it
    skeleton_image = skeletonize(image == 0)
    # Resized in Q2 already

    # Before saving, check if the directory exists
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save the blurred image
    cv2.imwrite(output_path, (skeleton_image).astype(np.uint8) * 255)

def process_images_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                # Find the relative path to preserve the directory
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
                process_image(input_path, output_path)

input_folder = 'Output_q2'
output_folder = 'Output_q3'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder(input_folder, output_folder)
