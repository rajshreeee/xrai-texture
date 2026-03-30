import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from skimage import io, color
import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import functional as F
import config
from utils import add_to_GLCM_results
from pathlib import Path
import segmentation_models_pytorch as smp

#Input data preprocess and load to the model starts

# Custom transform for resizing images
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)

# Load and preprocess the image
def preprocess_image(image_path, input_size=(512, 512)):
    transform = transforms.Compose([
        ResizeTransform(input_size),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


#image process and load to the models
#Activation feature maps for Layer 1 start

def get_feature_maps(model, input_image, model_choice):
    activations = []

    def hook(model, input, output):
        activations.append(output)

    # Access the underlying model if model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    if model_choice == 1 or model_choice == 2:
        # Register hook
        hook_handle = model.backbone.layer4.register_forward_hook(hook)
    elif model_choice == 5 or model_choice == 6:
        hook_handle = model.encoder.layer4.register_forward_hook(hook)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Remove the hook
    hook_handle.remove()

    return activations[0]

#Activation feature maps for Layer 1 end 


# Define a function to compute GLCM properties for a given image
def glcm_properties(image):
    # Convert image to integer type in the range [0, 255]
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Compute GLCM
    glcm = mh.features.haralick(image_uint8)

    # Extract desired GLCM properties
    properties = {
        "ASM": glcm.mean(axis=0)[0],
        "contrast": glcm.mean(axis=0)[1],
        "correlation": glcm.mean(axis=0)[2],
        "variance": glcm.mean(axis=0)[3],
        "IDM": glcm.mean(axis=0)[4],
        "sum_average": glcm.mean(axis=0)[5],
        "sum_entropy": glcm.mean(axis=0)[6],
        "entropy": glcm.mean(axis=0)[7],
        "diff_entropy": glcm.mean(axis=0)[8],
        "IMC1": glcm.mean(axis=0)[9],
        "IMC2": glcm.mean(axis=0)[10],
        "MCC": glcm.mean(axis=0)[11],
        "autocorrelation": glcm.mean(axis=0)[12]
    }
    
    return properties

# Compute GLCM properties for each feature map
def compute_glcm_properties(feature_maps, gray_image):
    epsilon = 1e-10
    num_feature_maps = feature_maps.size(1)
    original_glcm_properties = glcm_properties(gray_image)  # Compute GLCM properties for the original image

    # Initialize lists to store absolute differences for each GLCM property
    absolute_differences = {prop: [] for prop in original_glcm_properties}

    for i in range(num_feature_maps):
        feature_map = feature_maps[:, i:i+1, :, :]  # Extract a single feature map
        feature_map_numpy = feature_map.cpu().detach().numpy()[0, 0]  # Convert to numpy array
        
        # Compute GLCM properties for the feature map
        feature_map_glcm_properties = glcm_properties(feature_map_numpy)
        
        # Compute absolute differences for each GLCM property
        for prop in original_glcm_properties:
            original_value = original_glcm_properties[prop]
            feature_value = feature_map_glcm_properties[prop]
            absolute_difference = abs(original_value - feature_value ) / (0.5 * (abs(original_value) + abs(feature_value)) + epsilon)
            absolute_differences[prop].append(absolute_difference)

        # Print or compare GLCM properties between the original image and the feature map
        #print(f"GLCM Properties for Feature Map {i+1}:")
        #print("Original Image:", original_glcm_properties)
       # print("Feature Map:", feature_map_glcm_properties)
       # print()  # Add space between feature maps

    # Compute average absolute differences for each GLCM property
    average_absolute_differences = {prop: np.mean(absolute_differences[prop]) for prop in original_glcm_properties}

    # Sort GLCM properties based on average absolute differences
    sorted_properties = sorted(average_absolute_differences, key=average_absolute_differences.get)

    # Print feature ranking
    results = dict()
    for i, prop in enumerate(sorted_properties):
        results[prop] = average_absolute_differences[prop]

    return results

def load_fpn(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(model_path, map_location=device)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            print("Model loaded successfully as a full model.")
        else:
            raise ValueError("Loaded object is not a model. Trying as state_dict...")
    except Exception as e:
        print(f"Failed to load model directly: {e}")
        
        model = smp.FPN(
            encoder_name="resnet34",        # Choose encoder, e.g., resnet34, resnet50
            encoder_weights="imagenet",     # Use pretrained weights from ImageNet
            in_channels=3,                  # Input channels (e.g., RGB)
            classes=1                       # Number of output classes
        )
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            print("Model loaded successfully as state_dict.")
        except Exception as e:
            print(f"Failed to load model as state_dict: {e}")
            raise RuntimeError(f"Failed to load the model from {model_path}. Ensure the file and architecture match.")
    
    return model

def load_linknet(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(model_path, map_location=device)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            print("Model loaded successfully as a full model.")
        else:
            raise ValueError("Loaded object is not a model. Trying as state_dict...")
    except Exception as e:
        print(f"Failed to load model directly: {e}")
        
        model = smp.Linknet(
            encoder_name="resnet34",       # Choose encoder, e.g., resnet34, resnet50
            encoder_weights="imagenet",    # Use pretrained weights from ImageNet
            in_channels=3,                 # Input channels (e.g., RGB)
            classes=1                      # Number of output classes
        )
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            print("Model loaded successfully as state_dict.")
        except Exception as e:
            print(f"Failed to load model as state_dict: {e}")
            raise RuntimeError(f"Failed to load the model from {model_path}. Ensure the file and architecture match.")
    
    return model

def analyse_GLCM(model_choice, dataset_choice):

    if dataset_choice == 1:
        image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        
        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/CBIS_DDSM/CBIS_DDSM_DEEPLAB_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/CBIS_DDSM/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/CBIS_DDSM/CBIS_DDSM_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/CBIS_DDSM/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/CBIS_DDSM/CBIS_DDSM_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/CBIS_DDSM/Feature_10/fpn_segmentation.pth'
            model = load_fpn(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/CBIS_DDSM/CBIS_DDSM_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/CBIS_DDSM/Feature_10/linknet_segmentation.pth'
            model = load_linknet(model_path)

    elif dataset_choice == 2:
        image_path= config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_DEEPLAB_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/CBIS_DDSM_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/CBIS_DDSM_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/CBIS_DDSM_CLAHE/Feature_10/fpn_segmentation.pth'
            model = load_fpn(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/CBIS_DDSM_CLAHE/Feature_10/linknet_segmentation.pth'
            model = load_linknet(model_path)

    elif dataset_choice == 3:
        image_path= config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/HAM10000/HAM10000_Deeplab_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/HAM10000/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/HAM10000/HAM10000_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/HAM10000/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/HAM10000/HAM10000_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/HAM10000/Feature_10/fpn_segmentation.pth'
            model = load_fpn(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/HAM10000/HAM10000_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/HAM10000/Feature_10/linknet_segmentation.pth'
            model = load_linknet(model_path)

    elif dataset_choice == 4:
        image_path= config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/HAM10000_CLAHE/HAM10000_CLAHE_Deeplab_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/HAM10000_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/HAM10000_CLAHE/HAM10000_CLAHE_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/HAM10000_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/HAM10000_CLAHE/HAM10000_CLAHE_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/HAM10000_CLAHE/Feature_10/fpn_segmentation.pth'
            model = load_fpn(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/HAM10000_CLAHE/HAM10000_CLAHE_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/HAM10000_CLAHE/Feature_10/linknet_segmentation.pth'
            model = load_linknet(model_path)

    elif dataset_choice == 5:
        image_path= config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/POLYP/POLYP_Deeplab_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/POLYP/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/POLYP/POLYP_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/POLYP/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/POLYP/POLYP_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/POLYP/Feature_10/fpn_segmentation.pth'
            model = load_fpn(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/POLYP/POLYP_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/POLYP/Feature_10/linknet_segmentation.pth'
            model = load_linknet(model_path)

    else:
        image_path= config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/GLCM/Deeplab/POLYP_CLAHE/POLYP_CLAHE_Deeplab_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Deeplab/POLYP_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/GLCM/Fcn/POLYP_CLAHE/POLYP_CLAHE_FCN_GLCM.csv'
            model = torch.load(config.saved_models_path + '/Fcn/POLYP_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/GLCM/Fpn/POLYP_CLAHE/POLYP_CLAHE_FPN_GLCM.csv'
            model_path = config.saved_models_path + '/Fpn/POLYP_CLAHE/Feature_10/fpn_segmentation.pth'
            model = torch.load(model_path)
        elif model_choice == 6:
            result_path = config.results_path + '/GLCM/Linknet/POLYP_CLAHE/POLYP_CLAHE_LINKNET_GLCM.csv'
            model_path = config.saved_models_path + '/Linknet/POLYP_CLAHE/Feature_10/linknet_segmentation.pth'
            model = torch.load(model_path)

    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)

    input_image = preprocess_image(image_path)
    
    # Move the input tensor to GPU if available
    if torch.cuda.is_available():
        input_image = input_image.cuda()
        model = model.cuda()        

    # Apply the model
    model.eval()
    #with torch.no_grad():
        #output = model(input_image)['out']


    # Example: Get feature maps from 'layer4'
    feature_maps  = get_feature_maps(model, input_image, model_choice)

    # Compute and visualize GLCM properties for each feature map
    result = compute_glcm_properties(feature_maps, gray_image)

    
    os.makedirs(config.results_path + '/GLCM', exist_ok=True)
    os.makedirs(Path(result_path).parent, exist_ok=True)
    print(Path(result_path).parent)
    add_to_GLCM_results(result_path, result)
    print('Results are stored in ' + result_path)
