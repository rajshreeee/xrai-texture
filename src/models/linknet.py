import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import calculate_iou, add_to_test_results
import config
import segmentation_models_pytorch as smp


def train_linknet(save_path, data_loader):

    # LinkNet model setup
    model = smp.Linknet(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34, resnet50
        encoder_weights="imagenet",     # Use pretrained weights from ImageNet
        in_channels=3,                  # Input channels (RGB images)
        classes=1                       # Number of output classes
    )
    
    # Move the model to GPU if available
    if torch.cuda.is_available():
        device_id = [0,1]  # Choose the GPU device ID you want to use
        #model = nn.DataParallel(model, device_ids=device_id)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            model = nn.DataParallel(model , device_ids=device_id)
        torch.cuda.set_device(device_id[0])
        model = model.cuda()
    else:
        model = model.cpu()
        print("CUDA is not available. Training on CPU.")

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 20  # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_scores = []

        for batch_idx, (images, masks) in enumerate(data_loader):
            # Check for problematic batch
            if images.shape[0] < 2 or images.shape[2] < 2 or images.shape[3] < 2 or \
            masks.shape[0] < 2 or masks.shape[2] < 2 or masks.shape[3] < 2:
                print(f"Skipping batch {batch_idx} due to unexpected size. "
                    f"Image shape: {images.shape}, Mask shape: {masks.shape}")
                continue

            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)

            if masks.dim() == 4:
                masks = masks.squeeze(1)

            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iou_score = calculate_iou(torch.sigmoid(outputs), masks)
            iou_scores.append(iou_score)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], "
                    f"Loss: {loss.item():.4f}, IoU: {iou_score:.4f}")

        epoch_loss = running_loss / len(data_loader)
        epoch_iou = np.mean(iou_scores)
        print(f"End of Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Average IoU: {epoch_iou:.4f}")

    print('Finished Training')

    os.makedirs(save_path, exist_ok=True)
    torch.save(model, save_path + '/linknet_segmentation.pth')

def test_linknet(result_path, dataset, feature_dataset_choice, data_loader):

    print("Testing LinkNet on Feature: " + str(feature_dataset_choice) + " of " + dataset)
    # Assuming the model is initially loaded for CPU

    model_path = config.saved_models_path + '/Linknet/' + dataset + '/Feature_' + str(feature_dataset_choice) + '/linknet_segmentation.pth'

    if not os.path.exists(model_path):
        print("The given model does not exist, Train the model before testing.")
        return

    # Determine the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load the model
    try:
        # Attempt to load as a full model
        model = torch.load(model_path, map_location=device)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
        else:
            raise ValueError("Loaded object is not a model. Trying as state_dict...")
    except Exception as e:
        print(f"Failed to load model directly: {e}")
        
        model = smp.Linknet(
            encoder_name="resnet34",        # Example encoder
            encoder_weights="imagenet",    # Use pretrained weights
            in_channels=3,                 # RGB images
            classes=1                      # Single output class
        )
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
        except Exception as e:
            print(f"Failed to load model as state_dict: {e}")
            raise RuntimeError(f"Failed to load the model from {model_path}. Ensure the file and architecture match.")

    # Evaluate the model on the test dataset
    model.eval()
    iou_scores = []

    for images, masks in data_loader:
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)

        iou_score = calculate_iou(torch.sigmoid(outputs), masks)
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)
    print(average_iou)

    os.makedirs(config.results_path, exist_ok=True)
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)

    print(f"Testing Successful. Results added to " + result_path)