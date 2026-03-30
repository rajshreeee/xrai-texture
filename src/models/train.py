import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from utils import clear_screen
from models.fcn import train_fcn
from models.deeplabv3 import train_deeplab
from models.unet import train_unet
from models.hrnet import train_hrnet
from models.fpn import train_fpn
from models.linknet import train_linknet
from models.fcb_former_paper import train_fcbformer

import config
import cv2
import numpy as np
import time


def prompt_model():
    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")
    print("4. HR-Net")
    print("5. FPN-Net")
    print("6. Link-Net")
    print("7. FCBFormer")
    
    # return 7
    
    choice = 7
    
    while True:
            try:
                choice = int(input("Select Model (1-7): "))
                if 1 <= choice <= 7:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 7 available functions.")
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
    print("8. CBIS_DDSM_LAPLACIAN")
    print("9: Custom CBIS_DDSM") #Preprocessed by custom kernels
    

    choice = 7
    # return 7
    
    while True:
            try:
                choice = int(input("Select Dataset (1-9): "))
                if 1 <= choice <= 9:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 9 available datasets.")
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
    print("11. Custom Feature")

    choice = 10
    while True:
            try:
                choice = int(input("Select Dataset (1-11): "))
                if 1 <= choice <= 11:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 11 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")
    
    return choice

# Custom transform for resizing images
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)

class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        # work around for now in MacOS as it was reading .DS_Store file. Please update the list in case of any new file extension.
        valid_img_extensions = (".jpg", ".jpeg", ".png")
        self.images = [f for f in os.listdir(images_dir) if f.endswith(valid_img_extensions)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.split('.')[0] + '.png') #TODO:: REMOVE AFTER DEMO OR FIX
        
        # Read image and mask using cv2
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image loading was successful
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        # Check if mask loading was successful
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert mask to binary
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Convert images to PIL format
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # Apply transforms if provided
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

from torch.utils.data import DataLoader, random_split

#Image transform function for TorchVision vs Originals
def image_transforms(model=None):
    if model=='FCBFormer':
        image_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 256x256 for Swin Transformer
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform
    else: 
        image_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform
        
    

def create_data_loader(dataset_choice, feature_dataset_choice, validation_split=0.2, batch_size=4):
    """
    Create training and validation data loaders.

    Args:
        dataset_choice (int): Choice of dataset.
        feature_dataset_choice (int): Feature choice within the dataset.
        validation_split (float): Fraction of the dataset to use for validation.
        batch_size (int): Batch size for the data loaders.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """
    # Define image and mask transformations (ORIGINAL FROM INTAKE 3) - 
    #TODO::Uncomment this line for all other models apart from the new transformers in fcbformer (must explain to the team)
    # image_transform = transforms.Compose([
    #     ResizeTransform((512, 512)),  # Resize to 512x512
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    #For the torchvision models
    image_transform = transforms.Compose([
    ResizeTransform((512, 512)),  # Resize to 256x256 for Swin Transformer
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])

    
    mask_transform = transforms.Compose([
        ResizeTransform((512, 512)),  # Resize to 512x512
        transforms.ToTensor()
    ])

    # Set paths based on dataset choice
    images_dir = ''
    if dataset_choice == 1:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_dataset_path + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/train/masks'
    elif dataset_choice == 2:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_dataset_path + '/train/masks'
    elif dataset_choice == 3:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_dataset_path + '/train/images'
        else:
            images_dir = config.HAM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/train/masks'
    elif dataset_choice == 4:
        if feature_dataset_choice == 10:
            images_dir = config.HAM_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.HAM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.HAM_dataset_path + '/train/masks'
    elif dataset_choice == 5:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_dataset_path + '/train/images'
        else:
            images_dir = config.POLYP_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/train/masks'
    elif dataset_choice == 6:
        if feature_dataset_choice == 10:
            images_dir = config.POLYP_CLAHE_dataset_path + '/train/images'
        else:
            images_dir = config.POLYP_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.POLYP_dataset_path + '/train/masks'
    elif dataset_choice == 7:
        if feature_dataset_choice == 10:
            # images_dir = config.CBIS_DDSM_PATCHES + '/train/aug_images'
            images_dir = config.CBIS_DDSM_PATCHES + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_PATCHES + '/train/textures/Feature_' + str(feature_dataset_choice)
        # masks_dir = config.CBIS_DDSM_PATCHES + '/train/aug_masks'
        masks_dir = config.CBIS_DDSM_PATCHES + '/train/masks'

    elif dataset_choice == 8:
        if feature_dataset_choice == 10:
            images_dir = config.CBIS_DDSM_LAPLACIAN + '/train/images'
        else:
            images_dir = config.CBIS_DDSM_LAPLACIAN + '/train/textures/Feature_' + str(feature_dataset_choice)
        masks_dir = config.CBIS_DDSM_LAPLACIAN + '/train/masks'

    elif dataset_choice == 9:
        images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/deeplab-l4-k4/train/textures"
        masks_dir = config.CBIS_DDSM_PATCHES + '/train/masks'

    # Create full dataset
    dataset = CancerDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    # Split dataset into training and validation sets
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def get_images_dir(dataset_choice, feature_dataset_choice):

    # Create your datasets and data loaders
    images_dir = ''
    
    if dataset_choice == 1:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_dataset_path + '/train/images'
            test_images_dir = config.CBIS_DDSM_dataset_path + '/test/images'
        else:
            train_images_dir = config.CBIS_DDSM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_dataset_path + '/test/masks'
    elif dataset_choice == 2:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_CLAHE_dataset_path + '/test/masks'
    elif dataset_choice == 3:
        if feature_dataset_choice == 10:
            train_images_dir = config.HAM_dataset_path + '/train/images'
            test_images_dir = config.HAM_dataset_path + '/test/images'

        else:
            train_images_dir = config.HAM_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.HAM_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 4:
        if feature_dataset_choice == 10:
            train_images_dir = config.HAM_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.HAM_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.HAM_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.HAM_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.HAM_dataset_path + '/test/masks'
    elif dataset_choice == 5:
        if feature_dataset_choice == 10:
            train_images_dir = config.POLYP_dataset_path + '/train/images'
            test_images_dir = config.POLYP_dataset_path + '/test/images'

        else:
            train_images_dir = config.POLYP_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.POLYP_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.POLYP_dataset_path + '/test/masks'
    elif dataset_choice == 6:
        if feature_dataset_choice == 10:
            train_images_dir = config.POLYP_CLAHE_dataset_path + '/train/images'
            test_images_dir = config.POLYP_CLAHE_dataset_path + '/test/images'

        else:
            train_images_dir = config.POLYP_CLAHE_dataset_path + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.POLYP_CLAHE_dataset_path + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.POLYP_dataset_path + '/test/masks'
    elif dataset_choice == 7:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_PATCHES + '/train/images'
            test_images_dir = config.CBIS_DDSM_PATCHES + '/test/images'

        else:
            train_images_dir = config.CBIS_DDSM_PATCHES + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_PATCHES + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_PATCHES + '/test/masks'
    

    elif dataset_choice == 8:
        if feature_dataset_choice == 10:
            train_images_dir = config.CBIS_DDSM_LAPLACIAN + '/train/images'
            test_images_dir = config.CBIS_DDSM_LAPLACIAN + '/test/images'

        else:
            train_images_dir = config.CBIS_DDSM_LAPLACIAN + '/train/textures/Feature_' + str(feature_dataset_choice)
            test_images_dir = config.CBIS_DDSM_LAPLACIAN + '/test/textures/Feature_' + str(feature_dataset_choice)
        test_masks_dir = config.CBIS_DDSM_LAPLACIAN + '/test/masks'
    
    elif dataset_choice == 9:
        train_images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/deeplab-l4-k4/train/textures"
        test_images_dir = "/ediss_data/ediss2/xai-texture/src/preprocessing/results/deeplab-l4-k4/test/textures"
        test_masks_dir = config.CBIS_DDSM_PATCHES + '/test/masks'

    return train_images_dir,test_images_dir,test_masks_dir

def train_model():
    clear_screen() #clears the terminal screen
    
    #get model and dataset from user
    model_choice = prompt_model()
    clear_screen()
    dataset_choice = prompt_dataset()
    clear_screen()
    feature_dataset_choice = prompt_feature_dataset()

    data_loader, val_loader = create_data_loader(dataset_choice, feature_dataset_choice)
    train_images_dir, test_images_dir, mask_images_dir = get_images_dir(dataset_choice, feature_dataset_choice)

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
        dataset = 'CBIS_DDSM_LAPLACIAN'
    elif dataset_choice == 9:
        dataset = 'CUSTOM'
    start_time = time.time()
    if model_choice == 1:
        train_deeplab(config.saved_models_path + '/Deeplab/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 2:
        train_fcn(config.saved_models_path + '/Fcn/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 3:
        train_unet(config.saved_models_path + '/Unet/'+dataset+'/Feature_' + str(feature_dataset_choice), dataset, train_images_dir, test_images_dir, mask_images_dir)

    elif model_choice == 4:
        train_hrnet(config.saved_models_path + '/Hrnet/'+dataset+'/Feature_' + str(feature_dataset_choice), dataset, train_images_dir, test_images_dir, mask_images_dir)

    elif model_choice == 5:
        train_fpn(config.saved_models_path + '/Fpn/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)

    elif model_choice == 6:
        train_linknet(config.saved_models_path + '/Linknet/'+dataset+'/Feature_' + str(feature_dataset_choice), data_loader)
    
    elif model_choice == 7:
        train_fcbformer(save_path=config.saved_models_path + '/FCBFormer/' + dataset + '/intake5_run3' + str(feature_dataset_choice), data_loader=data_loader, val_loader=val_loader)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

    return