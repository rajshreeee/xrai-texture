import subprocess
import platform
import torch
import pandas as pd
import os
import numpy as np
import csv

#clear the terminal screen
def clear_screen():
    # Check if the operating system is Windows
    if platform.system() == "Windows":
        subprocess.run("cls", shell=True, check=True)
    else:
        # Assume the operating system is Unix/Linux/Mac
        subprocess.run("clear", shell=True, check=True)

# Helper function to calculate IoU
def calculate_iou(preds, labels):
    preds = preds > 0.5  # Threshold the predictions to get binary output
    labels = labels > 0.5
    intersection = (preds & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (preds | labels).float().sum((1, 2))         # Will be zero if both are 0

    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our division to avoid 0/0
    return torch.mean(iou).item()  # Return the mean IoU score for the batch

# Helper function to calculate IoU
def calculate_iou_unet(preds, labels):
    # Threshold the predicted and ground truth masks to obtain binary masks
    preds = (preds > 0).astype(np.uint8)
    labels = (labels > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    
    # Calculate IoU
    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our division to avoid division by zero
    return iou

# Helper function to calculate accuracy
def calculate_pixelaccuracy(preds, labels):
    # Convert predicted and ground truth masks to binary masks
    preds_binary = (preds > 0).astype(np.uint8)
    labels_binary = (labels > 0).astype(np.uint8)
    
    # Count the number of matching pixels
    correct_pixels = np.sum(preds_binary == labels_binary)
    
    # Calculate total number of pixels
    total_pixels = preds_binary.size
    
    # Calculate accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy


def add_to_test_results(file_path, dataset, feature_dataset_choice, value):

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
     # Check if the CSV file exists
    if os.path.isfile(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame(columns=['Dataset', 'IOU Accuracy'])
    
    # Check if the row to update exists
    if dataset in df['Dataset'].values:
        # Update the existing row
        df.update([dataset, value])
    else:
        # Append a new row
        df = df._append({'Dataset' : feature_dataset_choice, 'IOU Accuracy' : value}, ignore_index=True)
    
    # Write the DataFrame to CSV
    df.to_csv(file_path, index=False)

def add_to_test_results_unet(csv_path, dataset_index, iou_cancer, pixel_accuracy):
    # Check if CSV file exists
    file_exists = os.path.exists(csv_path)
    
    # Check if the file is empty
    file_empty = os.stat(csv_path).st_size == 0 if file_exists else True

    # If file doesn't exist, create a new CSV file with headers
    if not file_exists:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
    
    if file_empty:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', 'IOU Cancer', 'Pixel Accuracy'])

    # Read existing data from CSV file
    existing_data = []
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            existing_data.append(row)

    # Check if dataset index already exists
    dataset_found = False
    for i, row in enumerate(existing_data):
        if i > 0 and row[0] == str(dataset_index):
            # Update values if dataset index exists
            existing_data[i][1] = str(iou_cancer)
            existing_data[i][2] = str(pixel_accuracy)
            dataset_found = True
            break

    # If dataset index doesn't exist, append a new row
    if not dataset_found:
        existing_data.append([str(dataset_index), str(iou_cancer), str(pixel_accuracy)])

    # Write updated data back to CSV file
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)

    print("Data has been written to", csv_path)

def add_to_LTEM_results(file_path, dataset, values):
    
     # Check if the CSV file exists
    if os.path.isfile(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame(columns=['Dataset', 'Layer1', 'Layer2', 'Layer3', 'Layer4'])
    
    # Check if the row to update exists
    if dataset in df['Dataset'].values:
        # Update the existing row
        df.update([dataset, values[0], values[1], values[2], values[3]])
    else:
        # Append a new row
        df = df._append({'Dataset': dataset, 'Layer1': values[0], 'Layer2' : values[1], 'Layer3': values[2], 'Layer4': values[3]}, ignore_index=True)
    
    # Write the DataFrame to CSV
    df.to_csv(file_path, index=False)

def add_to_LTEM_unet_results(filepath, result_list):
    # Clear the contents of the CSV file (if it exists) at the beginning
    if os.path.exists(filepath):
        os.remove(filepath)

    # Open the CSV file in append mode
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        header = ['Index'] + [f'Layer_{i+1}' for i in range(len(result_list[0]))]
        writer.writerow(header)

        # Iterate over each array generated at different loops
        for idx, array in enumerate(result_list, start=1):
            # Write data row with the array elements
            writer.writerow([f'Feature_{idx}'] + array)

def add_to_GLCM_results(file_path, values):
     
    df = pd.DataFrame(columns=values.keys())
    df = df._append(values, ignore_index=True)
    
    # Write the DataFrame to CSV
    df.to_csv(file_path, index=False)

def add_to_GLCM_Unet_results(file_path, result_array):
    # Define the headers from the keys of the first dictionary
    headers = ["idx"] + list(result_array[0].keys())  # Add 'idx' as the first header

    # Write the array of dictionaries to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write headers
        writer.writeheader()
        
        # Write rows
        for idx, data in enumerate(result_array):
            # Add index to the dictionary
            data_with_idx = {'idx': 'Layer_'+str(idx)}  # Create a new dictionary with 'idx' key
            data_with_idx.update(data)     # Merge with original dictionary
            
            writer.writerow(data_with_idx)