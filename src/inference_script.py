import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Assuming the models are in the structure you've provided
from models.fcb_former import load_fcbformer, FCBFormer
from models.classification.ResNets_models import StackedEnsemble, load_resnet_model

def load_classification_model(model_path: str, device) -> torch.nn.Module:
    """
    Loads the trained StackedEnsemble classification model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model path not found: {model_path}")

    # Instantiate the model architecture
    # The base ResNets are loaded globally in ResNets_models.py
    model = StackedEnsemble()
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel prefix if it exists
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("Classification model loaded successfully.")
    return model.eval()

def preprocess_for_segmentation(image: Image.Image, input_size=512):
    """
    Prepares an image for the FCBFormer segmentation model.
    """
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet stats
    ])
    return transform(image.convert("RGB"))

def preprocess_for_classification(image: Image.Image, input_size=224):
    """
    Prepares an image for the ResNet-based classification model.
    """
    # Convert grayscale to 3-channel image as required by the model
    img_array = np.array(image.convert("L"))
    img_array_3_channel = np.stack([img_array] * 3, axis=-1)
    image_3_channel = Image.fromarray(img_array_3_channel)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # As defined in ResNets_models.py
    ])
    return transform(image_3_channel)

def decode_classification_output(output_tensor):
    """
    Decodes the output of the classification model into human-readable format.
    """
    # Output format: [pathology, mass_shape, birads_0, birads_1, ..., birads_5]
    
    # Pathology (index 0)
    pathology_pred = torch.sigmoid(output_tensor[0]).item() > 0.5
    pathology_label = "Malignant" if pathology_pred else "Benign"

    # Mass Shape (index 1)
    shape_classes = {0: "Round", 1: "Oval", 2: "Lobulated", 3: "Irregular"}
    shape_pred = torch.round(output_tensor[1]).int().item()
    shape_label = shape_classes.get(shape_pred, "Unknown")

    # BI-RADS (indices 2-7)
    birads_probs = torch.softmax(output_tensor[2:], dim=0)
    birads_pred = torch.argmax(birads_probs).item()

    return {
        "pathology": pathology_label,
        "mass_shape": shape_label,
        "bi_rads": birads_pred
    }

def run_inference(image_path: str, seg_model, class_model, device):
    """
    Runs the full inference pipeline: segmentation followed by classification.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    original_image = Image.open(image_path)

    # --- 1. Segmentation ---
    seg_input = preprocess_for_segmentation(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = seg_model(seg_input)
    
    # Generate a binary mask from the output
    seg_mask_prob = torch.sigmoid(seg_output)
    binary_mask = (seg_mask_prob > 0.5).squeeze(0).cpu().numpy()

    # --- 2. Classification ---
    class_input = preprocess_for_classification(original_image).unsqueeze(0).to(device)
    with torch.no_grad():
        class_output = class_model(class_input)

    # --- 3. Decode and Return Results ---
    classification_results = decode_classification_output(class_output.squeeze(0))
    
    return binary_mask, classification_results


    # --- Configuration ---
SEG_MODEL_PATH = "/ediss_data/ediss2/xai-texture/src/models/fcb_former.py"
CLASS_MODEL_PATH = "path/to/your/classification_model.pth" #TODO::Update this path with your final model for classification.
TEST_IMAGE_PATH = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_CLAHE/test/images/Mass-Training_P_00068_RIGHT_MLO_crop4.jpg"

# Check for model paths
if not os.path.exists(SEG_MODEL_PATH) or not os.path.exists(CLASS_MODEL_PATH):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
    print("!!! WARNING: Update model paths in the __main__ block. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Load Models ---
segmentation_model = load_fcbformer(SEG_MODEL_PATH)
classification_model = load_classification_model(CLASS_MODEL_PATH, DEVICE)

# --- Run Pipeline ---
try:
    segmentation_mask, classification_results = run_inference(
        image_path=TEST_IMAGE_PATH,
        seg_model=segmentation_model,
        class_model=classification_model,
        device=DEVICE
    )

    # --- Display Results ---
    print("\n--- Inference Results ---")
    print(f"Image: {TEST_IMAGE_PATH}")
    
    print("\n[Classification]")
    for key, value in classification_results.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")

    print("\n[Segmentation]")
    print(f"  - Mask shape: {segmentation_mask.shape}")
    print(f"  - Unique values in mask: {np.unique(segmentation_mask)}")

    # Optional: Save the segmentation mask as an image
    mask_img = Image.fromarray((segmentation_mask.squeeze() * 255).astype(np.uint8))
    mask_save_path = "segmentation_mask_output.png"
    mask_img.save(mask_save_path)
    print(f"  - Segmentation mask saved to: {mask_save_path}")

except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Please ensure the test image path is correct.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
