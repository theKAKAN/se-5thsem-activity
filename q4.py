# DO NOT RUN MANUALLY

import cv2
import numpy as np
import csv
import os
import pandas as pd

def save_pixel_intensity_to_csv(input_image_path, csv_writer):
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    pixel_values = np.asarray(image)
    my_list = pixel_values.flatten().tolist()
    my_list.insert(0,input_image_path)
    csv_writer.writerow(my_list)

input_folder = 'Output_q2'
output_excel_file = 'pixel_intensity_values.xlsx'
sheet_name = 'Output Q2'

column_names = [f'Pixel_{i}' for i in range(64*64)]
df = pd.DataFrame(columns=column_names)

with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    csv_writer = csv.writer(open('pixel_intensity_values.csv', 'w', newline=''))
    
    # Recursively process images in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                input_image_path = os.path.join(root, file)
                save_pixel_intensity_to_csv(input_image_path, csv_writer)
                
    # Read the CSV file and save the data to the Excel file
    df2 = pd.read_csv('pixel_intensity_values.csv', header=None)
    df2.to_excel(writer, sheet_name=sheet_name, startrow=len(df) + 1, header=False, index=False)
