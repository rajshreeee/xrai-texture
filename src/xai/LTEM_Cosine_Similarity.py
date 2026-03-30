import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import config
import torch.nn as nn
import segmentation_models_pytorch as smp

from pathlib import Path
import matplotlib.pyplot as plt

from utils import add_to_LTEM_results

# Define a function to load the models
def load_model(model_path):

    model = torch.load(model_path)  # Load model on CPU
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    return model

def load_model_cpu(model_path):

    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model on CPU
    model.eval()
    return model

# Define a function to get feature maps using hook
def get_feature_maps(model, input_image, layer_name, model_choice):
    activations = []

    def hook(model, input, output):
        activations.append(output)

    # Access the underlying model if model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    # Register hook
    if model_choice == 1 or model_choice == 2:
        target_layer = getattr(model.backbone, layer_name)
    elif model_choice == 5 or model_choice == 6:
        target_layer = getattr(model.encoder, layer_name)
    hook_handle = target_layer.register_forward_hook(hook)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_image)

    # Remove the hook
    hook_handle.remove()

    return activations[0]

def load_fpn(model_path):
    model = smp.FPN(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34, resnet50
        encoder_weights="imagenet",     # Use pretrained weights from ImageNet
        in_channels=3,                  # Input channels (RGB images)
        classes=1                       # Number of output classes
    )
    model.load_state_dict(torch.load(model_path))
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model

def load_linknet(model_path):
    model = smp.Linknet(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34, resnet50
        encoder_weights="imagenet",     # Use pretrained weights from ImageNet
        in_channels=3,                  # Input channels (RGB images)
        classes=1                       # Number of output classes
    )
    model.load_state_dict(torch.load(model_path))
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model

# Define a function to load the model (whether full model or state_dict)
def load_fpn_model(model_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if the checkpoint is a state_dict
    if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
        # If it's a state_dict, create a new model and load the state_dict into it
        model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        model.load_state_dict(checkpoint)
    else:
        # If it's a full model, we assume torch.load() returns the model directly
        model = checkpoint

    model.eval()
    return model

# Define a function to load the model (whether full model or state_dict)
def load_linknet_model(model_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Check if the checkpoint is a state_dict
    if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
        # If it's a state_dict, create a new model and load the state_dict into it
        model = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        model.load_state_dict(checkpoint)
    else:
        # If it's a full model, we assume torch.load() returns the model directly
        model = checkpoint

    model.eval()
    return model

def calculate_cosine_similarity(model_choice, dataset_choice, feature):

    if dataset_choice == 1:
        test_image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        
        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/CBIS_DDSM/CBIS_DDSM_DEEPLAB_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/CBIS_DDSM/CBIS_DDSM_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/CBIS_DDSM/CBIS_DDSM_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/CBIS_DDSM/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/CBIS_DDSM/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/CBIS_DDSM/CBIS_DDSM_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/CBIS_DDSM/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/CBIS_DDSM/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    elif dataset_choice == 2:
        test_image_path = config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_DEEPLAB_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM_CLAHE/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM_CLAHE/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/CBIS_DDSM_CLAHE/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/CBIS_DDSM_CLAHE/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/CBIS_DDSM_CLAHE/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/CBIS_DDSM_CLAHE/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    elif dataset_choice == 3:
        test_image_path = config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/HAM10000/HAM10000_Deeplab_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/HAM10000/HAM10000_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/HAM10000/HAM10000_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/HAM10000/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/HAM10000/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/HAM10000/HAM10000_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/HAM10000/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/HAM10000/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    elif dataset_choice == 4:
        test_image_path = config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/HAM10000_CLAHE/HAM10000_CLAHE_Deeplab_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000_CLAHE/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/HAM10000_CLAHE/HAM10000_CLAHE_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000_CLAHE/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/HAM10000_CLAHE/HAM10000_CLAHE_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/HAM10000_CLAHE/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/HAM10000_CLAHE/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/HAM10000_CLAHE/HAM10000_CLAHE_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/HAM10000_CLAHE/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/HAM10000_CLAHE/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    elif dataset_choice == 5:
        test_image_path = config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.POLYP_dataset_path + f'/test/textures/Feature_{i}/cju2suk42469908015ngmq6f2.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/POLYP/POLYP_Deeplab_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/POLYP/POLYP_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/POLYP/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/POLYP/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/POLYP/POLYP_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/POLYP/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/POLYP/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/POLYP/POLYP_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/POLYP/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/POLYP/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    else:
        test_image_path = config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.POLYP_CLAHE_dataset_path + f'/test/textures/Feature_{i}/cju2suk42469908015ngmq6f2.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/POLYP_CLAHE/POLYP_CLAHE_Deeplab_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP_CLAHE/Feature_'+ str(feature)+'/deeplab_v3_segmentation.pth')
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/POLYP_CLAHE/POLYP_CLAHE_FCN_LTEM.csv'
            model1 = load_model_cpu(config.saved_models_path + '/Fcn/POLYP_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
            model2 = load_model_cpu(config.saved_models_path + '/Fcn/POLYP_CLAHE/Feature_'+ str(feature)+'/fcn_resnet101_segmentation.pth')
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/POLYP_CLAHE/POLYP_CLAHE_FPN_LTEM.csv'
            model_path1 = config.saved_models_path + '/Fpn/POLYP_CLAHE/Feature_10/fpn_segmentation.pth'
            model1 = load_fpn_model(model_path1)
            model_path2 = config.saved_models_path + '/Fpn/POLYP_CLAHE/Feature_'+ str(feature)+'/fpn_segmentation.pth'
            model2 = load_fpn_model(model_path2)
        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/POLYP_CLAHE/POLYP_CLAHE_LINKNET_LTEM.csv'
            model_path1 = config.saved_models_path + '/Linknet/POLYP_CLAHE/Feature_10/linknet_segmentation.pth'
            model1 = load_linknet_model(model_path1)
            model_path2 = config.saved_models_path + '/Linknet/POLYP_CLAHE/Feature_'+ str(feature)+'/linknet_segmentation.pth'
            model2 = load_linknet_model(model_path2)

    # Prepare a test image
    #test_image_path = config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
    test_image = Image.open(test_image_path).convert("RGB")

    # Define a transform for the test image
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transform to the test image
    test_image = test_transform(test_image).unsqueeze(0)

    # Load both trained models

    
    #if model == 1:
    #    model1 = load_model(config.saved_models_path + '/Deeplab/Feature_10/deeplab_v3_segmentation.pth')
    #    model2 = load_model(config.saved_models_path + '/Deeplab/Feature_' + str(feature) + '/deeplab_v3_segmentation.pth')
    #    result_path = config.results_path + '/LTEM_Cosine_Similarity/Deeplab_LTEM.csv'

    #elif model == 2:
    #    model1 = load_model(config.saved_models_path + '/FCN/Feature_10/fcn_resnet101_epoch20_segmentation.pth')
    #    model2 = load_model(config.saved_models_path + '/FCN/Feature_' + str(feature) + '/fcn_resnet101_epoch20_segmentation.pth')
    #    result_path = config.results_path + '/LTEM_Cosine_Similarity/FCN_LTEM.csv'

    result = []

    # Plot the results
    # Compare the feature maps from different layers
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    for layer_name in layers:
        # Get feature maps from the specified layer of both models
        with torch.no_grad():
            features1 = get_feature_maps(model1, test_image, layer_name, model_choice).cpu().squeeze(0)
            features2 = get_feature_maps(model2, test_image, layer_name, model_choice).cpu().squeeze(0)

        # Calculate the similarities using cosine similarity
        similarities = []
        for i in range(features1.size(0)):  # Iterate over each feature map
            similarity = cosine_similarity(features1[i].reshape(1, -1), features2[i].reshape(1, -1))[0, 0]
            similarities.append(similarity)

        # Calculate the average similarity
        average_similarity = np.mean(similarities)
        result.append(average_similarity)

    os.makedirs(Path(result_path).parent, exist_ok=True)

    add_to_LTEM_results(result_path, feature, result)
    print("Results of comparison of Feature_10 model with Feature_" + str(feature) + " are stored in " + result_path)


def cosine_similarity_analysis_textureimages(model_choice, dataset_choice):

    if dataset_choice == 1:
        test_image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        
        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/CBIS_DDSM/CBIS_DDSM_DEEPLAB_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM/Feature_10/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/CBIS_DDSM/CBIS_DDSM_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM/Feature_10/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/CBIS_DDSM/CBIS_DDSM_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/CBIS_DDSM/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/CBIS_DDSM/CBIS_DDSM_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/CBIS_DDSM/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)

    elif dataset_choice == 2:
        test_image_path= config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
        texture_image_paths = [config.CBIS_DDSM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
        
        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_DEEPLAB_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/CBIS_DDSM_CLAHE/Feature_10/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/CBIS_DDSM_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/CBIS_DDSM_CLAHE/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/CBIS_DDSM_CLAHE/CBIS_DDSM_CLAHE_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/CBIS_DDSM_CLAHE/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)

    elif dataset_choice == 3:
        test_image_path = config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/HAM10000/HAM10000_Deeplab_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000/Feature_10/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/HAM10000/HAM10000_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000/Feature_10/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/HAM10000/HAM10000_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/HAM10000/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/HAM10000/HAM10000_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/HAM10000/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)

    elif dataset_choice == 4:
        test_image_path = config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'
        texture_image_paths = [config.HAM_CLAHE_dataset_path + f'/test/textures/Feature_{i}/ISIC_0024306.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/HAM10000_CLAHE/HAM10000_CLAHE_Deeplab_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/HAM10000_CLAHE/Feature_10/deeplab_v3_segmentation.pth')
        
        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/HAM10000_CLAHE/HAM10000_CLAHE_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/HAM10000_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')
        
        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/HAM10000_CLAHE/HAM10000_CLAHE_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/HAM10000_CLAHE/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/HAM10000_CLAHE/HAM10000_CLAHE_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/HAM10000_CLAHE/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)

    elif dataset_choice == 5:
        test_image_path = config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.POLYP_dataset_path + f'/test/textures/Feature_{i}/cju2suk42469908015ngmq6f2.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/POLYP/POLYP_Deeplab_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP/Feature_10/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/POLYP/POLYP_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/POLYP/Feature_10/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/POLYP/POLYP_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/POLYP/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/POLYP/POLYP_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/POLYP/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)

    else:
        test_image_path = config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'
        texture_image_paths = [config.POLYP_CLAHE_dataset_path + f'/test/textures/Feature_{i}/cju2suk42469908015ngmq6f2.jpg' for i in range(1, 10)]

        # Load the model and move it to GPU
        if model_choice == 1:
            result_path = config.results_path + '/LTEM/Deeplab/POLYP_CLAHE/POLYP_CLAHE_Deeplab_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Deeplab/POLYP_CLAHE/Feature_10/deeplab_v3_segmentation.pth')

        elif model_choice == 2:
            result_path = config.results_path + '/LTEM/Fcn/POLYP_CLAHE/POLYP_CLAHE_FCN_LTEM.png'
            model = load_model_cpu(config.saved_models_path + '/Fcn/POLYP_CLAHE/Feature_10/fcn_resnet101_segmentation.pth')

        elif model_choice == 5: 
            result_path = config.results_path + '/LTEM/Fpn/POLYP_CLAHE/POLYP_CLAHE_FPN_LTEM.png'
            model_path1 = config.saved_models_path + '/Fpn/POLYP_CLAHE/Feature_10/fpn_segmentation.pth'
            model = load_fpn_model(model_path1)

        elif model_choice == 6:
            result_path = config.results_path + '/LTEM/Linknet/POLYP_CLAHE/POLYP_CLAHE_LINKNET_LTEM.png'
            model_path1 = config.saved_models_path + '/Linknet/POLYP_CLAHE/Feature_10/linknet_segmentation.pth'
            model = load_linknet_model(model_path1)
    
    # Define the custom transform for resizing images
    class ResizeTransform:
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            return img.resize(self.size, Image.BILINEAR)

    # Define the image transformation pipeline
    image_transform = transforms.Compose([
        ResizeTransform((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the trained DeepLabv3 model
    model.eval()  # Set the model to evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    # Hook function to extract feature maps
    feature_maps = {}

    def hook_fn(module, input, output, name):
        feature_maps[name] = output

    # Register hooks to specific layers of the model
    target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    hooks = []
    
    if model_choice == 1 or model_choice == 2:
        # Register hook
        for name, layer in model.backbone.named_modules():
            if name in target_layers:
                hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))
                print(f"Hook registered for layer: {name}")
    elif model_choice == 5 or model_choice == 6:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        for name, layer in model.encoder.named_modules():
            if name in target_layers:
                hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))
                print(f"Hook registered for layer: {name}")

    # Load and preprocess the test image
    test_image = Image.open(test_image_path).convert("RGB")
    input_image = image_transform(test_image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        input_image = input_image.cuda()

    # Forward pass through the model to extract feature maps of the test image
    with torch.no_grad():
        _ = model(input_image)

    # Remove hooks after extracting feature maps
    for hook in hooks:
        hook.remove()

    # Define a function to compute the cosine similarity
    def compute_cosine_similarity(feature_map, texture_feature_map):
        similarities = []
        for i in range(feature_map.size(1)):
            fm = feature_map[0, i].cpu().detach().numpy().flatten()
            tex_fm = texture_feature_map[0, i].cpu().detach().numpy().flatten()
            similarity = cosine_similarity(fm.reshape(1, -1), tex_fm.reshape(1, -1))
            similarities.append(similarity[0][0])
        return similarities

    # Initialize a dictionary to store the average similarities
    similarity_scores = {layer: [] for layer in target_layers}

    for i, texture_path in enumerate(texture_image_paths):
        texture_image = Image.open(texture_path).convert("RGB")
        texture_image = image_transform(texture_image).unsqueeze(0)  # Add batch dimension

        if torch.cuda.is_available():
            texture_image = texture_image.cuda()

        texture_feature_maps = {}

        # Hook function to extract feature maps for texture images
        def hook_fn_texture(module, input, output, name):
            texture_feature_maps[name] = output

        # Register hooks for texture image feature map extraction
        hooks_texture = []
        
        if model_choice == 1 or model_choice == 2:
            # Register hook
            for name, layer in model.backbone.named_modules():
                if name in target_layers:
                    hooks_texture.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn_texture(module, input, output, name)))
                    print(f"Hook registered for layer: {name}")
        elif model_choice == 5 or model_choice == 6:
            for name, layer in model.encoder.named_modules():
                if name in target_layers:
                    hooks_texture.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn_texture(module, input, output, name)))
                    print(f"Hook registered for layer: {name}")

        # Forward pass through the model to extract feature maps of the texture image
        with torch.no_grad():
            _ = model(texture_image)

        # Remove hooks after extracting feature maps
        for hook in hooks_texture:
            hook.remove()

        print(f"\nProcessing texture image {i+1}")

        for layer_name, feature_map in feature_maps.items():
            texture_feature_map = texture_feature_maps[layer_name]
            similarities = compute_cosine_similarity(feature_map, texture_feature_map)
            avg_similarity = np.mean(similarities)
            similarity_scores[layer_name].append(avg_similarity)
            print(f"Layer {layer_name} - Texture {i+1} Average Cosine Similarity: {avg_similarity}")

    os.makedirs(Path(result_path).parent, exist_ok=True)

    # Plot the results
    plt.figure(figsize=(15, 8))

    for layer_name, similarities in similarity_scores.items():
        plt.plot(range(1, len(texture_image_paths) + 1), similarities, label=layer_name)

    plt.xlabel('Texture Image')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Average Cosine Similarity by Layer and Texture Image')
    plt.legend()
    plt.xticks(range(1, len(texture_image_paths) + 1))
    plt.savefig(Path(result_path), format='png', dpi=300, bbox_inches='tight')
    plt.show()

def cosine_similarity_analysis(model, dataset):
    print("Conducting Cosine Similarity Analysis...")
    for i in range(1, 10):
        calculate_cosine_similarity(model, dataset, i)
    cosine_similarity_analysis_textureimages(model, dataset)