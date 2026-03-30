import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from utils import clear_screen
from models.fcn import test_fcn
from models.deeplabv3 import test_deeplab
from models.unet import test_unet
from models.hrnet import test_hrnet
from models.fpn import test_fpn
from models.linknet import test_linknet
from models.fcb_former_paper import test_fcbformer
import config

def prompt_model():
    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")
    print("4. HR-Net")
    print("5. FPN-Net")
    print("6. Link-Net")
    print("7. FCBFormer")

    choice = None
    choice =7
    return choice
    
    while True:
            try:
                choice = int(input("Select Model (1-7): "))
                if 1 <= choice <= 7:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_dataset():

    print("1. CBIS_DDSM")
    print("2. CBIS_DDSM_CLAHE")
    print("3. HAM10000")
    print("4. HAM10000_CLAHE")
    print("5. POLYP")
    print("6. POLYP_CLAHE")
    print("7. CBIS_DDSM_PATCHES")
    print("8. CUSTOM")

    choice = None
    choice = 8
    return choice
    
    while True:
            try:
                choice = int(input("Select Dataset (1-8): "))
                if 1 <= choice <= 8:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available datasets.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_feature_dataset():

    print("1. Feature 1 (L5E5 / E5L5)")
    print("2. Feature 2 (L5S5 / S5L5)")
    print("3. Feature 3 (L5R5 / L5R5)")
    print("4. Feature 4 (E5S5 / S5E5)")
    print("5. Feature 5 (E5R5 / R5E5)")
    print("6. Feature 6 (R5S5 / S5R5)")
    print("7. Feature 7 (S5S5)")
    print("8. Feature 8 (E5E5)")
    print("9. Feature 9 (R5R5)")
    print("10. Original")
    print("11. Custom")
    
    choice = None
    choice= 11
    return choice
    while True:
            try:
                choice = int(input("Select Dataset (1-11): "))
                if 1 <= choice <= 11:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 11 available texture dataset.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

# Custom transform for resizing images
# class ResizeTransform:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img):
#         return F.resize(img, self.size)

class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Filter out invalid files
        self.images = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) and f != ".DS_Store"
        ]
        self.images.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)

        # Handle mask with different possible formats
        mask_name = os.path.splitext(image_name)[0] + ".png"  # Change this if masks have different extensions
        mask_path = os.path.join(self.masks_dir, mask_name)

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Skipping {image_name} due to an error: {e}")
            return self.__getitem__((idx + 1) % len(self.images))  # Move to next valid image safely

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, image_name

class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)

# class CancerDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_transform = image_transform
#         self.mask_transform = mask_transform

#         # Valid image filtering
#         self.images = [
#             f for f in os.listdir(images_dir)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) and f != ".DS_Store"
#         ]
#         self.images.sort()

#         # Define L5 and E5 filters for L5E5 and E5L5
#         L5 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)
#         E5 = np.array([-1, -2, 0, 2, 1]).reshape(1, 5)
#         self.L5E5_kernel = np.outer(L5, E5)
#         self.E5L5_kernel = np.outer(E5, L5)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_name = self.images[idx]
#         image_path = os.path.join(self.images_dir, image_name)
#         mask_name = os.path.splitext(image_name)[0] + ".jpg"
#         mask_path = os.path.join(self.masks_dir, mask_name)

#         try:
#             # Read image using OpenCV in grayscale
#             image_bgr = cv2.imread(image_path)
#             if image_bgr is None:
#                 raise FileNotFoundError(f"Image not found: {image_path}")
#             gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

#             # Apply L5E5 and E5L5, then average
#             l5e5 = cv2.filter2D(gray, -1, self.L5E5_kernel)
#             e5l5 = cv2.filter2D(gray, -1, self.E5L5_kernel)
#             texture = ((l5e5 + e5l5) / 2).astype(np.uint8)

#             # Convert to 3-channel RGB for compatibility with image transforms
#             texture_rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
#             image = Image.fromarray(texture_rgb)

#             # Load mask
#             mask = Image.open(mask_path).convert("L")

#         except Exception as e:
#             print(f"Skipping {image_name} due to error: {e}")
#             return self.__getitem__((idx + 1) % len(self))

#         if self.image_transform:
#             image = self.image_transform(image)
#         if self.mask_transform:
#             mask = self.mask_transform(mask)

#         return image, mask, image_name

def create_data_loader(dataset_choice, feature_dataset_choice):
    
    image_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # image_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # Masks: Match the model's output size
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Should match image size
        transforms.ToTensor()
    ])

    # Create your datasets and data loaders
    images_dir = ''
    if dataset_choice == 1:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_dataset_path + '/test/images'
        else:
            images_dir = config.CBIS_DDSM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/test/masks'
    elif dataset_choice == 2:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/images'
        else:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/test/masks'
    elif dataset_choice == 3:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_dataset_path + '/test/images'
        else:
            images_dir = config.HAM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 4:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_CLAHE_dataset_path + '/test/images'
        else:
            images_dir = config.HAM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 5:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_dataset_path + '/test/images'
        else:
            images_dir = config.POLYP_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/test/masks'
    elif dataset_choice == 6:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_CLAHE_dataset_path + '/test/images'
        else:
            images_dir = config.POLYP_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/test/masks'
    elif dataset_choice == 7:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_PATCHES + '/test/images'
        else:
            images_dir = config.CBIS_DDSM_PATCHES + '/test/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_PATCHES + '/test/masks'

    elif dataset_choice == 8:
        images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/kernel0-kr/test/textures"
        masks_dir = config.CBIS_DDSM_PATCHES + '/test/masks'

    dataset = CancerDataset(
        images_dir= images_dir,
        masks_dir= masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    return DataLoader(dataset, batch_size=4, shuffle=True, drop_last = True)

def get_images_dir(dataset_choice, feature_dataset_choice):
    """
    Retrieve paths for training images, testing images, and testing masks based on dataset and feature choices.
    """
    dataset_paths = {
        1: config.CBIS_DDSM_dataset_path,
        2: config.CBIS_DDSM_CLAHE_dataset_path,
        3: config.HAM_dataset_path,
        4: config.HAM_CLAHE_dataset_path,
        5: config.POLYP_dataset_path,
        6: config.POLYP_CLAHE_dataset_path,
        7: config.CBIS_DDSM_PATCHES
    }
    
    if dataset_choice != 8 and dataset_choice not in dataset_paths:
        raise ValueError(f"Invalid dataset_choice: {dataset_choice}")

    if dataset_choice == 8:
        train_images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/kernel0-kr/train/textures"
        test_images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/kernel0-kr/test/textures"
        test_masks_dir = config.CBIS_DDSM_PATCHES + '/test/masks'
    else:
        base_path = dataset_paths[dataset_choice]
        train_dir_suffix = '/train/images' if feature_dataset_choice == 10 else f'/train/textures/Feature_{feature_dataset_choice}'
        test_dir_suffix = '/test/images' if feature_dataset_choice == 10 else f'/test/textures/Feature_{feature_dataset_choice}'
        test_masks_suffix = '/test/masks'
    
        train_images_dir = base_path + train_dir_suffix
        test_images_dir = base_path + test_dir_suffix
        test_masks_dir = base_path + test_masks_suffix
    
    return train_images_dir, test_images_dir, test_masks_dir


def test_model():
    clear_screen() #clears the terminal screen
    
    #get model and dataset from user
    model_choice = prompt_model()
    clear_screen()
    dataset_choice = prompt_dataset()
    clear_screen()
    feature_dataset_choice = prompt_feature_dataset()

    data_loader = create_data_loader(dataset_choice, feature_dataset_choice)
    train_images_dir, test_images_dir, test_masks_dir = get_images_dir(dataset_choice, feature_dataset_choice)

    if dataset_choice == 1:
        dataset = 'CBIS_DDSM'
    elif dataset_choice == 2:
        dataset = 'CBIS_DDSM_CLAHE'
    elif dataset_choice == 3:
        dataset = 'HAM10000'
    elif dataset_choice == 4:
        dataset = 'HAM10000_CLAHE'
    elif dataset_choice == 5:
        dataset = 'POLYP'
    elif dataset_choice == 6:
        dataset = 'POLYP_CLAHE'
    elif dataset_choice == 7:
        dataset = 'CBIS_DDSM_PATCHES'
    elif dataset_choice == 8:
        dataset = 'CUSTOM'
    if model_choice == 1:
        test_deeplab(config.results_path + "/Deeplab/" + dataset + "/Feature_" + str(feature_dataset_choice) + '/Deeplab_test.csv', dataset, feature_dataset_choice, data_loader)

    elif model_choice == 2:
        test_fcn(config.results_path + "/FCN/" + dataset + "/Feature_" + str(feature_dataset_choice) + '/Fcn_test.csv', dataset, feature_dataset_choice, data_loader)

    elif model_choice == 3:
        test_unet(config.results_path + "/Unet/" + dataset + "/Feature_" + str(feature_dataset_choice) + '/Unet_test.csv', dataset, feature_dataset_choice,  test_images_dir, test_masks_dir)

    elif model_choice == 4:
        test_hrnet(config.results_path + "/Hrnet/" + dataset + "/Feature_" + str(feature_dataset_choice) + '/Hrnet_test.csv', dataset, feature_dataset_choice,  test_images_dir, test_masks_dir)

    elif model_choice == 5:
        test_fpn(config.results_path + "/Fpn/" + dataset + "/Feature_" + str(feature_dataset_choice) + '/Fpnet_test.csv', dataset, feature_dataset_choice,  data_loader)

    elif model_choice == 6:
        test_linknet(config.results_path + "/Linknet/" + dataset + "Feature_" + str(feature_dataset_choice) + '/Linknet_test.csv', dataset, feature_dataset_choice,  data_loader)
    
    elif model_choice == 7:
        test_fcbformer(config.results_path + "/FCBFormer/" + dataset + "TestFeature_" + str(feature_dataset_choice) + '/FCBFormer_test.csv', dataset, feature_dataset_choice,  data_loader)

    return