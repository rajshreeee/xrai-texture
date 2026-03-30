import torch
from mmseg.apis import init_model, inference_model
import mmcv
import matplotlib.pyplot as plt
import cv2
import config
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import pandas as pd

from mmseg.models import build_segmentor
from mmengine import Config
import config
import numpy as np
import mahotas as mh
from skimage import io, color
from utils import add_to_GLCM_Unet_results, add_to_GLCM_results
import os

def prompt_dataset():

    print("1. CBIS_DDSM")
    print("2. CBIS_DDSM_CLAHE")
    print("3. HAM10000")
    print("4. HAM10000_CLAHE")
    print("5. POLYP")
    print("6. POLYP_CLAHE")
    

    choice = None
    while True:
            try:
                choice = int(input("Select Dataset (1-6): "))
                if 1 <= choice <= 6:
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

    choice = None
    while True:
            try:
                choice = int(input("Select Dataset (1-10): "))
                if 1 <= choice <= 10:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 10 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def GLCM(feature_maps):

    # Compute GLCM properties for each feature map
    feature_map_glcm_properties = []
    for i in range(feature_maps.size(1)):
        feature_map = feature_maps[0, i].cpu().detach().numpy()
        glcm_properties = compute_glcm_properties(feature_map)
        feature_map_glcm_properties.append(glcm_properties)

    # Visualize GLCM properties for each feature map
    for i, glcm_properties in enumerate(feature_map_glcm_properties):
        print(f"GLCM Properties for Feature Map {i+1}:")
        for j, prop in enumerate(glcm_properties.mean(axis=0)):
            print(f"Feature {j+1}: {prop}")
            
            
# Define a function to compute GLCM properties for a given image
def compute_glcm_properties(image):
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

# Compute and visualize GLCM properties for each feature map
def compute_and_visualize_glcm_properties_firstapproach(feature_maps, gray_image):
    epsilon = 1e-10
    num_feature_maps = feature_maps.size(1)
    original_glcm_properties = compute_glcm_properties(gray_image)  # Compute GLCM properties for the original image

    # Initialize lists to store absolute differences for each GLCM property
    absolute_differences = {prop: [] for prop in original_glcm_properties}

    for i in range(num_feature_maps):
        feature_map = feature_maps[:, i:i+1, :, :]  # Extract a single feature map
        feature_map_numpy = feature_map.cpu().detach().numpy()[0, 0]  # Convert to numpy array
        
        # Compute GLCM properties for the feature map
        feature_map_glcm_properties = compute_glcm_properties(feature_map_numpy)
        
        # Compute absolute differences for each GLCM property
        for prop in original_glcm_properties:
            original_value = original_glcm_properties[prop]
            feature_value = feature_map_glcm_properties[prop]
            absolute_difference = abs(original_value - feature_value ) / (0.5 * (abs(original_value) + abs(feature_value)) + epsilon)
            absolute_differences[prop].append(absolute_difference)

        # Print or compare GLCM properties between the original image and the feature map
        # print(f"GLCM Properties for Feature Map {i+1}:")
        # print("Original Image:", original_glcm_properties)
        # print("Feature Map:", feature_map_glcm_properties)
        # print()  # Add space between feature maps

    # Compute average absolute differences for each GLCM property
    average_absolute_differences = {prop: np.mean(absolute_differences[prop]) for prop in original_glcm_properties}

    # Sort GLCM properties based on average absolute differences
    sorted_properties = sorted(average_absolute_differences, key=average_absolute_differences.get)

    # Print feature ranking
    print("GLCM Feature Ranking:")

    # Print feature ranking
    results = dict()
    for i, prop in enumerate(sorted_properties):
        results[prop] = average_absolute_differences[prop]
    
    return results, sorted_properties, average_absolute_differences
        

def preprocess_image(image_path):
    # Move the model to GPU if available
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # Load the image from the file path
    input_image = Image.open(image_path).convert("RGB")
    
    # Define the image transformations
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 512x512 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # Normalize the image tensor
    ])
    
    # Apply transformations and move the image to GPU
    input_image = image_transform(input_image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension
    input_image = input_image.to(device)
    return input_image

def get_feature_maps(model, input_image):
    layers_of_interest = [model.backbone.encoder[0], model.backbone.encoder[1], model.backbone.encoder[2], model.backbone.encoder[3], model.backbone.encoder[4], model.backbone.decoder[0], model.backbone.decoder[1], model.backbone.decoder[2], model.backbone.decoder[3]]  # Add more layers as needed
    
    # Initialize an empty dictionary to store the activations
    activations_dict = {layer.__class__.__name__: [] for layer in layers_of_interest}
    
    # Define the hook function
    def hook_fn(module, input, output):
        activations_dict[module.__class__.__name__].append(output)
    
    # Register the hook function to the desired layers
    for layer in layers_of_interest:
        # Register hook
        hook_handle = layer.register_forward_hook(hook_fn)
        
        # Perform forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_image)
    
        # Remove the hook
        hook_handle.remove()
    
    # Now `activations_dict` contains activations for each layer keyed by their names
    # You can access them like this:
    for layer_name, activations_list in activations_dict.items():
        print(layer_name)
        for i, activations in enumerate(activations_list):
            print(f"Activation {i + 1}:", activations.shape)
    
    featuremaps = [activations_dict['Sequential'][0], activations_dict['Sequential'][1], activations_dict['Sequential'][2], activations_dict['Sequential'][3], activations_dict['Sequential'][4],activations_dict['UpConvBlock'][0],activations_dict['UpConvBlock'][1],activations_dict['UpConvBlock'][2],activations_dict['UpConvBlock'][3]]
    featuremap_names = ['Encoder[0]', 'Encoder[1]', 'Encoder[2]', 'Encoder[3]', 'Encoder[4]', 'Decoder[0]', 'Decoder[1]', 'Decoder[2]', 'Decoder[3]' ]
    return featuremaps, featuremap_names

def get_feature_maps_hrnet(model, input_image):
    layers_of_interest = [model.backbone.conv1, model.backbone.bn1, model.backbone.conv2, model.backbone.bn2, model.backbone.relu, model.backbone.layer1[0], model.backbone.layer1[1], model.backbone.layer1[2], model.backbone.layer1[3], model.backbone.transition1[0], model.backbone.transition1[1], model.backbone.stage2[0], model.backbone.transition2[2], model.backbone.stage3[0], model.backbone.stage3[1], model.backbone.stage3[2], model.backbone.stage3[3], model.backbone.transition3[3], model.backbone.stage4[0], model.backbone.stage4[1], model.backbone.stage4[2]]  # Add more layers as needed
    
    # Initialize an empty dictionary to store the activations
    activations_dict = {layer.__class__.__name__: [] for layer in layers_of_interest}
    
    # Define the hook function
    def hook_fn(module, input, output):
        activations_dict[module.__class__.__name__].append(output)
    
    # Register the hook function to the desired layers
    for layer in layers_of_interest:
        # Register hook
        hook_handle = layer.register_forward_hook(hook_fn)
        
        # Perform forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_image)
    
        # Remove the hook
        hook_handle.remove()
    
    # Now `activations_dict` contains activations for each layer keyed by their names
    # You can access them like this:
    for layer_name, activations_list in activations_dict.items():
        print(layer_name)
        for i, activations in enumerate(activations_list):
            print(f"Activation {i + 1}:", activations)
    
    featuremaps = [activations_dict['Conv2d'][0], activations_dict['Conv2d'][1], activations_dict['SyncBatchNorm'][0], 
    activations_dict['SyncBatchNorm'][1], activations_dict['ReLU'][0], activations_dict['ReLU'][1], 
    activations_dict['Bottleneck'][0], activations_dict['Bottleneck'][1], activations_dict['Bottleneck'][2], 
    activations_dict['Bottleneck'][3], activations_dict['Sequential'][0], activations_dict['Sequential'][1], 
    activations_dict['Sequential'][2], activations_dict['Sequential'][3], activations_dict['HRModule'][0][0], 
    activations_dict['HRModule'][0][1], activations_dict['HRModule'][1][0], activations_dict['HRModule'][1][1], 
    activations_dict['HRModule'][1][2], activations_dict['HRModule'][2][0], activations_dict['HRModule'][2][1], 
    activations_dict['HRModule'][2][2], activations_dict['HRModule'][3][0], activations_dict['HRModule'][3][1],
    activations_dict['HRModule'][3][2], activations_dict['HRModule'][4][0], activations_dict['HRModule'][4][1], 
    activations_dict['HRModule'][4][2], activations_dict['HRModule'][5][0], activations_dict['HRModule'][5][1], 
    activations_dict['HRModule'][5][2], activations_dict['HRModule'][5][3], activations_dict['HRModule'][6][0], 
    activations_dict['HRModule'][6][1], activations_dict['HRModule'][6][2], activations_dict['HRModule'][6][3], 
    activations_dict['HRModule'][7][0], activations_dict['HRModule'][7][1], activations_dict['HRModule'][7][2], 
    activations_dict['HRModule'][7][3]]    

    featuremap_names = [
    'Conv2d[0]', 'Conv2d[1]',
    'SyncBatchNorm[0]', 'SyncBatchNorm[1]',
    'ReLU[0]', 'ReLU[1]',
    'Bottleneck[0]', 'Bottleneck[1]', 'Bottleneck[2]', 'Bottleneck[3]',
    'Sequential[0]', 'Sequential[1]', 'Sequential[2]', 'Sequential[3]',
    'HRModule[0][0]', 'HRModule[0][1]',
    'HRModule[1][0]', 'HRModule[1][1]', 'HRModule[1][2]',
    'HRModule[2][0]', 'HRModule[2][1]', 'HRModule[2][2]',
    'HRModule[3][0]', 'HRModule[3][1]', 'HRModule[3][2]',
    'HRModule[4][0]', 'HRModule[4][1]', 'HRModule[4][2]',
    'HRModule[5][0]', 'HRModule[5][1]', 'HRModule[5][2]', 'HRModule[5][3]',
    'HRModule[6][0]', 'HRModule[6][1]', 'HRModule[6][2]', 'HRModule[6][3]',
    'HRModule[7][0]', 'HRModule[7][1]', 'HRModule[7][2]', 'HRModule[7][3]']
    return featuremaps, featuremap_names

def save_last_layer_glcm_to_csv(glcm_results, result_path):
    """
    Save the last layer's GLCM results to a CSV file, preserving the original computed order.
    """
    properties = list(glcm_results.keys())  # Keep the computed order
    values = list(glcm_results.values())  # Corresponding values
    
    df = pd.DataFrame([values], columns=properties)  # Keep order intact
    
    os.makedirs(Path(result_path).parent, exist_ok=True)
    
    df.to_csv(result_path, index=False)

    print("Last layer's GLCM results saved to:", result_path)

def analyze_GLCM_Unet(choice_dataset):

    if choice_dataset == 1:
        dataset = 'CBIS_DDSM'
        image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    elif choice_dataset == 2:
        dataset = 'CBIS_DDSM_CLAHE'
        image_path= config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    elif choice_dataset == 3:
        dataset = 'HAM10000'
        image_path= config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'

    elif choice_dataset == 4:
        dataset = 'HAM10000_CLAHE'
        image_path= config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'

    elif choice_dataset == 5:
        dataset = 'POLYP'
        image_path= config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'

    elif choice_dataset == 6:
        dataset = 'POLYP_CLAHE'
        image_path= config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'


    config_file = config_file = config.saved_models_path + '/Unet/'+dataset+'/Feature_10/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
    checkpoint_file = config.saved_models_path + '/Unet/'+dataset+'/Feature_10/iter_16000.pth'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

    # Load the model using the configuration file
    cfg = Config.fromfile(config_file)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #image_path= config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    image = io.imread(image_path)

    gray_image = color.rgb2gray(image)

    input_image = preprocess_image(image_path)

    featuremaps, featuremap_names = get_feature_maps(model,input_image)
    #all_results = []
    #for i in range(len(featuremaps)):
       #print("First Approach GLCM comparison for Layer_"+str(i))
        #all_results.append(compute_and_visualize_glcm_properties_firstapproach(featuremaps[i],gray_image))

        
        #print(all_results[i])
    #os.makedirs(config.results_path + '/GLCM', exist_ok=True)
    #result_path = config.results_path + '/GLCM/Unet_GLCM_Feature_'+str(choice)+'_dataset.csv'
    #add_to_GLCM_Unet_results(result_path,all_results)
    
    #result_path = config.results_path + '/GLCM/Unet/'+dataset+'/'+dataset+'_UNET_GLCM.csv'
    #os.makedirs(Path(result_path).parent, exist_ok=True)
    #results = compute_and_visualize_glcm_properties_firstapproach(featuremaps[8],gray_image)
    #print(results)
    #add_to_GLCM_results(result_path, results)

    # Initialize list to store results
    results = []
    for i in range(len(featuremaps)):
        print(featuremap_names[i])
        result, sorted_properties, average_distances = compute_and_visualize_glcm_properties_firstapproach(featuremaps[i], gray_image)
        prop_list = [(prop, average_distances[prop]) for prop in sorted_properties]
        results.append([featuremap_names[i], prop_list])
        #print("Results: ", results)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=["FeatureMap", "GLCM_Property_Distances"])
    result_path = config.results_path + '/GLCM/Unet/'+dataset+'/'+dataset+'_Unet_GLCM.xlsx'
    os.makedirs(Path(result_path).parent, exist_ok=True)
    # Save DataFrame to an Excel file
    df_results.to_excel(result_path, index=False)
    print("Results saved to ", result_path)

    last_featuremap = featuremaps[-1]  # Take the last layer
    last_featuremap_name = featuremap_names[-1]  # Get the layer name
    result_path_csv = config.results_path + '/GLCM/Unet/'+dataset+'/'+dataset+'_Unet_GLCM_LastLayer.csv'

    print(f"Computing GLCM for last layer: {last_featuremap_name}")
    glcm_results, _, _ = compute_and_visualize_glcm_properties_firstapproach(last_featuremap, gray_image)

    save_last_layer_glcm_to_csv(glcm_results, result_path_csv)

    print(f"Saved last layer GLCM results for {dataset}.")


def analyze_GLCM_Hrnet(choice_dataset):

    if choice_dataset == 1:
        dataset = 'CBIS_DDSM'
        image_path= config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    elif choice_dataset == 2:
        dataset = 'CBIS_DDSM_CLAHE'
        image_path= config.CBIS_DDSM_CLAHE_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    elif choice_dataset == 3:
        dataset = 'HAM10000'
        image_path= config.HAM_dataset_path + '/test/images/ISIC_0024306.jpg'

    elif choice_dataset == 4:
        dataset = 'HAM10000_CLAHE'
        image_path= config.HAM_CLAHE_dataset_path + '/test/images/ISIC_0024306.jpg'

    elif choice_dataset == 5:
        dataset = 'POLYP'
        image_path= config.POLYP_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'

    elif choice_dataset == 6:
        dataset = 'POLYP_CLAHE'
        image_path= config.POLYP_CLAHE_dataset_path + '/test/images/cju2suk42469908015ngmq6f2.jpg'


    config_file = config_file = config.saved_models_path + '/Hrnet/'+dataset+'/Feature_10/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'
    checkpoint_file = config.saved_models_path + '/Hrnet/'+dataset+'/Feature_10/iter_16000.pth'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

    # Load the model using the configuration file
    cfg = Config.fromfile(config_file)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #image_path= config.patch_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'

    image = io.imread(image_path)

    gray_image = color.rgb2gray(image)

    input_image = preprocess_image(image_path)

    featuremaps, featuremap_names = get_feature_maps_hrnet(model,input_image)
    #all_results = []
    #for i in range(len(featuremaps)):
       #print("First Approach GLCM comparison for Layer_"+str(i))
        #all_results.append(compute_and_visualize_glcm_properties_firstapproach(featuremaps[i],gray_image))

        
        #print(all_results[i])
    #os.makedirs(config.results_path + '/GLCM', exist_ok=True)
    #result_path = config.results_path + '/GLCM/Unet_GLCM_Feature_'+str(choice)+'_dataset.csv'
    #add_to_GLCM_Unet_results(result_path,all_results)
    #result_path = config.results_path + '/GLCM/Hrnet/'+dataset+'/'+dataset+'_HRNET_GLCM.csv'
    #os.makedirs(Path(result_path).parent, exist_ok=True)

    # Initialize list to store results
    results = []
    for i in range(len(featuremaps)):
        print(featuremap_names[i])
        result, sorted_properties, average_distances = compute_and_visualize_glcm_properties_firstapproach(featuremaps[i], gray_image)
        prop_list = [(prop, average_distances[prop]) for prop in sorted_properties]
        results.append([featuremap_names[i], prop_list])
        #print("Results: ", results)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=["FeatureMap", "GLCM_Property_Distances"])
    result_path = config.results_path + '/GLCM/Hrnet/'+dataset+'/'+dataset+'_HRNET_GLCM.xlsx'
    os.makedirs(Path(result_path).parent, exist_ok=True)
    # Save DataFrame to an Excel file
    df_results.to_excel(result_path, index=False)
    print("Results saved to ", result_path)

    last_featuremap = featuremaps[-1]  # Take the last layer
    last_featuremap_name = featuremap_names[-1]  # Get the layer name
    result_path_csv = config.results_path + '/GLCM/Hrnet/'+dataset+'/'+dataset+'_HRNET_GLCM_LastLayer.csv'

    print(f"Computing GLCM for last layer: {last_featuremap_name}")
    glcm_results, _, _ = compute_and_visualize_glcm_properties_firstapproach(last_featuremap, gray_image)

    save_last_layer_glcm_to_csv(glcm_results, result_path_csv)

    print(f"Saved last layer GLCM results for {dataset}.")
