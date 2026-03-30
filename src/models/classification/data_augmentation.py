import cv2
import os
from PIL import Image
import numpy as np
from datetime import datetime

input_dir = '/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/masks'
output_dir = '/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masks'
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join('aug_log_train_images_masks.txt')

def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, 'a') as log_file:
        log_file.write(f"{timestamp} {message}\n")

# CLAHE setup
clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

def apply_clahe(img, clahe):
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        return clahe.apply(img)

def rotate(img, angle):
    pil_img = Image.fromarray(img)
    return np.array(pil_img.rotate(angle, expand=True))

log(f"Starting augmentation from {input_dir} to {output_dir}")
image_count = 0

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    filepath = os.path.join(input_dir, fname)
    base_name = os.path.splitext(fname)[0]

    try:
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    except Exception as e:
        log(f"Error reading {fname}: {e}")
        continue

    log(f"Processing: {fname}")
    image_count += 1

    for angle in [0, 90, 180, 270]:
        rot_img = rotate(img, angle)

        # Save plain rotated image
        rot_fname = f"{base_name}#rot{angle}.jpg"
        rot_path = os.path.join(output_dir, rot_fname)
        cv2.imwrite(rot_path, cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR))
        log(f"Saved: {rot_fname}")

        # Save CLAHE 1
        clahe1_img = apply_clahe(rot_img, clahe1)
        clahe1_fname = f"{base_name}#clahe1_rot{angle}.jpg"
        clahe1_path = os.path.join(output_dir, clahe1_fname)
        cv2.imwrite(clahe1_path, cv2.cvtColor(clahe1_img, cv2.COLOR_RGB2BGR))
        log(f"Saved: {clahe1_fname}")

        # Save CLAHE 2
        clahe2_img = apply_clahe(rot_img, clahe2)
        clahe2_fname = f"{base_name}#clahe2_rot{angle}.jpg"
        clahe2_path = os.path.join(output_dir, clahe2_fname)
        cv2.imwrite(clahe2_path, cv2.cvtColor(clahe2_img, cv2.COLOR_RGB2BGR))
        log(f"Saved: {clahe2_fname}")

log(f"Augmentation complete! Processed {image_count} images.")