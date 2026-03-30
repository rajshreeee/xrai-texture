import cv2
import os
import numpy as np

# Define the L5 and E5 masks
L5 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)
E5 = np.array([-1, -2, 0, 2, 1]).reshape(1, 5)

# Compute the 2D kernel for L5E5
L5E5_kernel = np.outer(L5, E5)

# Function to apply L5E5 convolution
def apply_L5E5_filter(image):
    return cv2.filter2D(image, -1, L5E5_kernel)

# Define input and output directories
input_folder = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/images"
output_folder = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/L5E5_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        filtered_image = apply_L5E5_filter(image)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, filtered_image)
        print(f"Processed and saved: {filename}")
    else:
        print(f"Failed to load: {filename}")

print("L5E5 texture filtering completed.")
