import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import calculate_iou, add_to_test_results  # Importing necessary utilities
import config  # Importing configuration parameters

def train_fcn(dataset, data_loader):

    # FCN model setup
    model = models.segmentation.fcn_resnet101(pretrained=True, progress=True)  # Loading pre-trained FCN-ResNet101 model
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))  # Modifying classifier head for 1 class segmentation

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
        print("CUDA is not available. Training on CPU.")

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Using Adam optimizer with learning rate 1e-4
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits

    # Training loop
    num_epochs = 20  # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        iou_scores = []

        for batch_idx, (images, masks) in enumerate(data_loader):  # Iterate over batches in the data loader
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()  # Move images and masks to GPU if available

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)['out']  # Forward pass

            # Squeeze the model's output to remove the channel dimension (if present)
            outputs = outputs.squeeze(1)

            # Ensure that the target masks have a single channel
            if masks.dim() == 4:
                masks = masks[:, 0, :, :]  # Take the first channel

            loss = criterion(outputs, masks.float())  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss
            iou_score = calculate_iou(torch.sigmoid(outputs), masks)  # Calculate IoU score
            iou_scores.append(iou_score)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], "
                    f"Loss: {loss.item():.4f}, IoU: {iou_score:.4f}")  # Print progress

        epoch_loss = running_loss / len(data_loader)  # Calculate average loss for the epoch
        epoch_iou = np.mean(iou_scores)  # Calculate average IoU for the epoch
        print(f"End of Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Average IoU: {epoch_iou:.4f}")  # Print epoch summary

    print('Finished Training')  # Training complete
    
    # Save the trained model
    os.makedirs(dataset, exist_ok=True)
    torch.save(model, dataset + '/fcn_resnet101_segmentation.pth')

def test_fcn(result_path, dataset, feature_dataset_choice, data_loader):
    print("Testing FCN on Feature: " + str(feature_dataset_choice) + " of " + dataset )

    model_path = config.saved_models_path + '/Fcn/' + dataset + '/Feature_' + str(feature_dataset_choice) + '/fcn_resnet101_segmentation.pth'
    print("Loaded model from " + model_path)
    if not os.path.exists(model_path):
        print("The given model does not exist, Train the model before testing.")
        return
    # Load the downloaded model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Evaluate the model on the test dataset
    model.eval()  # Set the model to evaluation mode
    iou_scores = []

    # Data details
    print("Number of test images:", len(data_loader.dataset))
    print("Batch size:", data_loader.batch_size)

    for images, masks in data_loader:  # Iterate over batches in the data loader
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()  # Move images and masks to GPU if available

        with torch.no_grad():
            outputs = model(images)['out']  # Forward pass without gradient calculation

        iou_score = calculate_iou(torch.sigmoid(outputs), masks)  # Calculate IoU score
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)  # Calculate average IoU score

    os.makedirs(config.results_path, exist_ok=True)
    add_to_test_results(result_path, dataset, feature_dataset_choice, average_iou)  # Add results to test results file

    print(f"Testing Successful. Results added to " + result_path)  # Print completion message
