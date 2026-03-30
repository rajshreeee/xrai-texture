# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
#from mmseg.registry import RUNNERS
from mmseg.registry import DATASETS
from mmseg.apis import init_model, inference_model
import cv2

from mmseg.datasets import BaseSegDataset
import config
from utils import calculate_iou, add_to_test_results_unet, calculate_iou_unet, calculate_pixelaccuracy


import os
from PIL import Image

# Function to convert mask image pixel values
def convert_mask(mask_path, output_path):
    mask = Image.open(mask_path)
    mask = mask.point(lambda p: p // 255)  # Convert pixel value from 255 to 1
    mask.save(output_path)

# Function to list mask image names in a text file
def list_mask_images(directory, output_file):
    mask_images = [os.path.splitext(mask_image)[0] for mask_image in os.listdir(directory) if mask_image.endswith('.png')]
    # Open the output file for writing
    with open(output_file, 'w') as f:
        for i, mask_image in enumerate(mask_images):
            # Write the image name; add a newline only if it's not the last image
            if i < len(mask_images) - 1:
                f.write(mask_image + '\n')
            else:
                f.write(mask_image)

def preprocess_dataset(dataset):

    # Paths
    train_masks_dir = osp.join("../data/" + dataset + '/train/masks')

    test_masks_dir = osp.join("../data/" + dataset + '/test/masks')
    segmentation_class_dir = osp.join("../data/" + dataset + '/SegmentationClass')
    train_txt_file = osp.join("../data/" + dataset + '/train.txt')
    val_txt_file = osp.join("../data/" + dataset + '/val.txt')

    # Create SegmentationClass folder if not exists
    if not os.path.exists(segmentation_class_dir):
        os.makedirs(segmentation_class_dir)

        # Process train masks
        for mask_image in os.listdir(train_masks_dir):
            if mask_image.endswith('.png'):
                mask_path = os.path.join(train_masks_dir, mask_image)
                output_path = os.path.join(segmentation_class_dir, mask_image)
                convert_mask(mask_path, output_path)

        # List train mask image names in train.txt
        list_mask_images(train_masks_dir, train_txt_file)

        # Process test masks
        for mask_image in os.listdir(test_masks_dir):
            if mask_image.endswith('.png'):
                mask_path = os.path.join(test_masks_dir, mask_image)
                output_path = os.path.join(segmentation_class_dir, mask_image)
                convert_mask(mask_path, output_path)

    # List test mask image names in val.txt
    list_mask_images(test_masks_dir, val_txt_file)
    return segmentation_class_dir, train_txt_file, val_txt_file


def train_hrnet(saved_model_path, dataset, train_images_dir, test_images_dir, mask_images_dir):
    segmentation_class_dir, train_txt_file, val_txt_file = preprocess_dataset(dataset)
    classes=('background', 'cancer')
    palette=[[0, 0, 0], [128, 128, 128]]
    
    @DATASETS.register_module()
    class StanfordBackgroundDataset(BaseSegDataset):
      METAINFO = dict(classes = classes, palette = palette)
      def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
    
    cfg = config.hrnet_config_file_path
    
    cfg = Config.fromfile(cfg)
    launcher = 'pytorch'
    
    cfg.train_dataloader = dict(
        batch_size=16,
        dataset=dict(
            ann_file=os.path.join(dataset,'train.txt'),
            data_prefix=dict(
                # Change the img_path and seg_map_path for different feature dataset accordingly
                img_path=train_images_dir, seg_map_path=segmentation_class_dir),
            data_root=config.data_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    scale=(
                        256,
                        256,
                    ),
                    type='RandomResize'),
                dict(
                    cat_max_ratio=0.75, crop_size=(
                        256,
                        256,
                    ), type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs'),
            ],
            type='StanfordBackgroundDataset'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=True, type='InfiniteSampler'))
    
    cfg.val_dataloader = dict(
        batch_size=1,
        dataset=dict(
            ann_file=os.path.join(dataset,'val.txt'),
            data_prefix=dict(
              # Change the img_path and seg_map_path for different feature dataset accordingly
                img_path=test_images_dir, seg_map_path=segmentation_class_dir),
            data_root=config.data_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=False, scale=(
                    256,
                    256,
                ), type='Resize'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
            type='StanfordBackgroundDataset'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=False, type='DefaultSampler'))
    cfg.train_cfg = dict(max_iters=16000, type='IterBasedTrainLoop', val_interval=4000)
    
    cfg.test_dataloader = cfg.val_dataloader
    
    cfg.default_hooks = dict(
        checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'))
    cfg.dataset_type = 'StanfordBackgroundDataset'
    cfg.work_dir = saved_model_path
    
    #Train
    runner = Runner.from_cfg(cfg)
    runner.train()


def test_hrnet(result_path, dataset, feature_dataset_choice, test_images_dir, mask_images_dir):
   # List all files in the input directory
   image_files = os.listdir(test_images_dir)
   config_file = config.saved_models_path + "/Hrnet/" + dataset + '/Feature_' + str(feature_dataset_choice)+'/fcn_hr18_4xb2-160k_cityscapes-512x1024.py'
   checkpoint_file = config.saved_models_path + '/Hrnet/' + dataset + '/Feature_' + str(feature_dataset_choice)+'/iter_16000.pth'
   device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
   model = init_model(config_file, checkpoint_file, device=device)
   total_iou = 0.0
   total_images = 0
   total_accuracy = 0.0
   # Process each image
   for image_file in image_files:
       # Skip non-image files
       if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
           continue
       
       # Construct the path to the input image
       input_image_path = os.path.join(test_images_dir, image_file)
       
       # Forward pass the input image through the model
       result = inference_model(model, input_image_path)
       
       # Convert predicted segmentation mask tensor to numpy array
       numpy_image = result.pred_sem_seg.data[0].cpu().numpy()
       
       # Load ground truth segmentation mask
       gt_mask_path = os.path.join(mask_images_dir, os.path.splitext(image_file)[0] + '.png')
       gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

       gt_mask[gt_mask == 255] = 1
       # Calculate IoU for the current image
       iou = calculate_iou_unet(numpy_image, gt_mask)

       acc = calculate_pixelaccuracy(numpy_image,gt_mask)
       
       # Update total IoU and total number of images
       total_iou += iou
       total_accuracy += acc
       total_images += 1


   # Calculate mean IoU
   if total_images > 0:
       average_iou = total_iou / total_images
       print("Mean IoU:", average_iou)
       mean_accuracy = total_accuracy / total_images
       print("Mean Accuracy:", mean_accuracy)
       add_to_test_results_unet(result_path, 'Feature_'+str(feature_dataset_choice), average_iou, mean_accuracy)

   else:
       print("No images found for evaluation.")
       
   print(f"Testing Successful. Results added to " + result_path)
