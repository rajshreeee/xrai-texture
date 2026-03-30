import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import calculate_iou, add_to_test_results
import config

def train_deeplab(save_path, data_loader):

    # DeepLabv3 model setup
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # Change the classifier head for 1 class
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # Also change the aux classifier

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
            outputs = model(images)['out']
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
    torch.save(model, save_path + '/deeplab_v3_segmentation.pth')

def test_deeplab(result_path, dataset, feature_dataset_choice, data_loader):

    print("Testing Deeplab on Feature: " + str(feature_dataset_choice) + " of " + dataset)
    # Assuming the model is initially loaded for CPU

    model_path = config.saved_models_path + '/Deeplab/' + dataset + '/Feature_' + str(feature_dataset_choice) + '/deeplab_v3_segmentation.pth'

    if not os.path.exists(model_path):
        print("The given model does not exist, Train the model before testing.")
        return

    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate the model on the test dataset
    model.eval()
    iou_scores = []

    for images, masks in data_loader:
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()

        with torch.no_grad():
            outputs = model(images)['out']

        iou_score = calculate_iou(torch.sigmoid(outputs), masks)
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)
    print(average_iou)

    #this crashes if directory is not present.
    os.makedirs(config.results_path, exist_ok=True)
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)

    print(f"Testing Successful. Results added to " + result_path)