import os
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from models.fcb_former import FCBFormer  # Adjust import as needed


class BranchInspector:
    """Hook-based inspector to capture intermediate outputs from each branch"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks on TB, FCB, and PH"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                    'output_shape': output.shape,
                }
            return hook
        
        # Hook TB (Transformer Branch)
        if hasattr(self.model, 'TB'):
            h = self.model.TB.register_forward_hook(get_activation('TB'))
            self.hooks.append(h)
        
        # Hook FCB (Fully Convolutional Branch)
        if hasattr(self.model, 'FCB'):
            h = self.model.FCB.register_forward_hook(get_activation('FCB'))
            self.hooks.append(h)
        
        # Hook PH (Prediction Head)
        if hasattr(self.model, 'PH'):
            h = self.model.PH.register_forward_hook(get_activation('PH'))
            self.hooks.append(h)
        
        # Hook up_tosize
        if hasattr(self.model, 'up_tosize'):
            h = self.model.up_tosize.register_forward_hook(get_activation('up_tosize'))
            self.hooks.append(h)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_results(self):
        """Return captured activations"""
        return self.activations


def inspect_single_model(model_path, input_size=224, verbose=True):
    """
    Inspect a single model with detailed branch outputs
    """
    
    if verbose:
        print(f"\nInspecting: {os.path.basename(model_path)}")
        print("=" * 80)
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k[7:] if k.startswith("module.") else k: v 
                     for k, v in state_dict.items()}
        
        model = FCBFormer(size=512)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Create inspector
        inspector = BranchInspector(model)
        inspector.register_hooks()
        
        # Forward pass
        test_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            final_output = model(test_input)
        
        # Get activations
        activations = inspector.get_results()
        inspector.remove_hooks()
        
        if verbose:
            print(f"\n📥 INPUT: {test_input.shape}")
            print("-" * 80)
            
            if 'TB' in activations:
                print(f"🔄 TB (Transformer Branch):")
                print(f"   Input:  {activations['TB']['input_shape']}")
                print(f"   Output: {activations['TB']['output_shape']}")
            
            if 'FCB' in activations:
                print(f"\n🔄 FCB (Fully Convolutional Branch):")
                print(f"   Input:  {activations['FCB']['input_shape']}")
                print(f"   Output: {activations['FCB']['output_shape']}")
            
            if 'up_tosize' in activations:
                print(f"\n⬆️  UPSAMPLING:")
                print(f"   Input:  {activations['up_tosize']['input_shape']}")
                print(f"   Output: {activations['up_tosize']['output_shape']}")
            
            if 'PH' in activations:
                print(f"\n🎯 PH (Prediction Head):")
                print(f"   Input:  {activations['PH']['input_shape']}")
                print(f"   Output: {activations['PH']['output_shape']}")
            
            print(f"\n📤 FINAL OUTPUT: {final_output.shape}")
            print("=" * 80)
        
        return {
            'model_name': os.path.basename(model_path),
            'model_path': model_path,
            'input_size': input_size,
            'input_shape': str(test_input.shape),
            'TB_input': str(activations.get('TB', {}).get('input_shape', 'N/A')),
            'TB_output': str(activations.get('TB', {}).get('output_shape', 'N/A')),
            'FCB_input': str(activations.get('FCB', {}).get('input_shape', 'N/A')),
            'FCB_output': str(activations.get('FCB', {}).get('output_shape', 'N/A')),
            'upsample_input': str(activations.get('up_tosize', {}).get('input_shape', 'N/A')),
            'upsample_output': str(activations.get('up_tosize', {}).get('output_shape', 'N/A')),
            'PH_input': str(activations.get('PH', {}).get('input_shape', 'N/A')),
            'PH_output': str(activations.get('PH', {}).get('output_shape', 'N/A')),
            'final_output': str(final_output.shape),
            'status': 'Success'
        }
        
    except Exception as e:
        return {
            'model_name': os.path.basename(model_path),
            'model_path': model_path,
            'input_size': input_size,
            'status': 'Failed',
            'error': str(e)
        }


def scan_all_models(base_path, test_sizes=[224, 352, 512], output_csv="model_branch_analysis.csv"):
    """
    Scan all models and analyze their branches
    """
    model_files = []
    
    # Find all .pth files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    
    print(f"Found {len(model_files)} model files")
    print("=" * 80)
    
    all_results = []
    
    for model_path in tqdm(model_files, desc="Inspecting models"):
        for size in test_sizes:
            result = inspect_single_model(model_path, input_size=size, verbose=False)
            all_results.append(result)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    return df


def main():
    """
    Main function - configure and run inspection
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect FCBFormer model branches')
    parser.add_argument('--base_path', type=str, 
                       default='/ediss_data/ediss2/xai-texture/saved_models/FCBFormer/CBIS_DDSM/Feature_1',
                       help='Base path to scan for models')
    parser.add_argument('--single_model', type=str, default=None,
                       help='Path to single model to inspect (optional)')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input size for single model inspection')
    parser.add_argument('--output_csv', type=str, default='model_branch_analysis.csv',
                       help='Output CSV filename')
    parser.add_argument('--test_sizes', nargs='+', type=int, default=[224],
                       help='List of input sizes to test')
    
    args = parser.parse_args()
    
    print("BRANCH-BY-BRANCH MODEL INSPECTION")
    print("=" * 80)
    print()
    
    if args.single_model:
        # Inspect single model
        print(f"Mode: Single Model Inspection")
        print(f"Model: {args.single_model}")
        print(f"Input Size: {args.input_size}×{args.input_size}")
        
        result = inspect_single_model(args.single_model, input_size=args.input_size, verbose=True)
        
        # Save single result
        df = pd.DataFrame([result])
        df.to_csv('single_model_inspection.csv', index=False)
        print(f"\n✅ Result saved to: single_model_inspection.csv")
        
    else:
        # Scan all models
        print(f"Mode: Batch Scanning")
        print(f"Base Path: {args.base_path}")
        print(f"Test Sizes: {args.test_sizes}")
        print()
        
        df = scan_all_models(args.base_path, test_sizes=args.test_sizes, output_csv=args.output_csv)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Group by model and show results
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            print(f"\n📁 {model_name}")
            print("-" * 40)
            
            for _, row in model_df.iterrows():
                if row['status'] == 'Success':
                    print(f"  {row['input_size']}×{row['input_size']}: "
                          f"TB={row['TB_output']}, FCB={row['FCB_output']}, Out={row['final_output']}")
                else:
                    print(f"  {row['input_size']}×{row['input_size']}: ❌ {row.get('error', 'Failed')[:50]}")


if __name__ == "__main__":
    main()
