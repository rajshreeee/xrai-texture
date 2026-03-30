import torch
import torch.nn as nn

# Import architecture variants
from models.fcb_former_current import FCBFormer as FCBFormer_PVT
from models.fcb_former import FCBFormer as FCBFormer_ViT

# model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM_PATCHES/Feature_10/fcbformer_best.pth"
# model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM/Feature_1/fcbformer_best.pth"
model_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM/Feature_1/fcbformer_segmentation.pth"
print(f"Inspecting: {model_path}")
print("=" * 80)

# Load checkpoint
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = {k[7:] if k.startswith("module.") else k: v 
             for k, v in state_dict.items()}

# Try loading with different architectures
architectures = [
    (FCBFormer_PVT, 'PVTv2'),
    (FCBFormer_ViT, 'ViT'),
]

for arch_class, arch_name in architectures:
    for model_size in [224, 352, 512]:
        try:
            model = arch_class(size=model_size)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            print(f"\n✅ LOADED: {arch_name} with size={model_size}")
            print("=" * 80)
            
            # Test with different input sizes
            for input_size in [224, 352, 512]:
                # Dictionary to capture outputs
                outputs = {}
                
                def get_hook(name):
                    def hook(module, input, output):
                        outputs[name] = output.shape
                    return hook
                
                # Register hooks
                h1 = model.TB.register_forward_hook(get_hook('TB'))
                h2 = model.FCB.register_forward_hook(get_hook('FCB'))
                
                with torch.no_grad():
                    test_input = torch.randn(1, 3, input_size, input_size)
                    final_output = model(test_input)
                
                # Remove hooks
                h1.remove()
                h2.remove()
                
                print(f"\nInput: {input_size}×{input_size}")
                print(f"  TB output:  {outputs.get('TB', 'N/A')}")
                print(f"  FCB output: {outputs.get('FCB', 'N/A')}")
                print(f"  Final:      {final_output.shape}")
            
            print("=" * 80)
            break  # Found working architecture
            
        except Exception as e:
            continue

print("\nDone!")
