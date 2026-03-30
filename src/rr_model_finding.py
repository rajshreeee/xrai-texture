import os
import torch
from tqdm import tqdm

# Import all architecture variants
from models.fcb_former import FCBFormer as FCBFormer_ViT
from models.fcb_former_current import FCBFormer as FCBFormer_PVT
from models.fcb_former_new_transformers import FCBFormer as FCBFormer_Swin


class DetailedInspector:
    """Capture TB and FCB intermediate outputs"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input': input[0].shape if isinstance(input, tuple) else input.shape,
                    'output': output.shape
                }
            return hook
        
        if hasattr(self.model, 'TB'):
            self.hooks.append(self.model.TB.register_forward_hook(get_activation('TB')))
        
        if hasattr(self.model, 'FCB'):
            self.hooks.append(self.model.FCB.register_forward_hook(get_activation('FCB')))
        
        if hasattr(self.model, 'up_tosize'):
            self.hooks.append(self.model.up_tosize.register_forward_hook(get_activation('upsample')))
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
    
    def get_results(self):
        return self.activations


def comprehensive_model_test(base_path):
    """
    Test all models with all architectures and check branch resolutions
    """
    architectures = [
        (FCBFormer_ViT, 'fcb_former (ViT)'),
        (FCBFormer_PVT, 'fcb_former_current (PVTv2)'),
        (FCBFormer_Swin, 'fcb_former_new_transformers (Swin)'),
    ]
    
    model_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    
    print(f"Testing {len(model_files)} models with all architectures")
    print("=" * 80)
    print("Looking for: 512×512 input with proper TB/FCB branch outputs")
    print("=" * 80)
    
    all_results = []
    paper_candidates = []
    
    for model_path in tqdm(model_files, desc="Testing models"):
        model_name = os.path.basename(model_path)
        
        for arch_class, arch_name in architectures:
            for model_size in [224, 352, 512]:
                try:
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    state_dict = {k[7:] if k.startswith("module.") else k: v 
                                 for k, v in state_dict.items()}
                    
                    # Load model
                    model = arch_class(size=model_size)
                    model.load_state_dict(state_dict, strict=True)
                    model.eval()
                    
                    # Test with 512×512 (paper spec)
                    inspector = DetailedInspector(model)
                    inspector.register_hooks()
                    
                    with torch.no_grad():
                        test_input = torch.randn(1, 3, 512, 512)
                        output = model(test_input)
                    
                    activations = inspector.get_results()
                    inspector.remove_hooks()
                    
                    # Extract shapes
                    tb_out = activations.get('TB', {}).get('output', None)
                    fcb_out = activations.get('FCB', {}).get('output', None)
                    
                    result = {
                        'model_name': model_name,
                        'architecture': arch_name,
                        'model_size': model_size,
                        'input': '512×512',
                        'TB_output': str(tb_out) if tb_out else 'N/A',
                        'FCB_output': str(fcb_out) if fcb_out else 'N/A',
                        'final_output': str(output.shape),
                        'success': True
                    }
                    
                    # Check if matches paper requirements
                    # Paper uses 512×512 input and output
                    # TB should produce reasonable resolution (not 28×28!)
                    # FCB should be at 512×512 or high resolution
                    
                    is_good_match = False
                    reason = []
                    
                    if output.shape == torch.Size([1, 1, 512, 512]):
                        reason.append("✅ Output: 512×512")
                        
                        # Check TB output - should NOT be 28×28 (that's wrong!)
                        # Good TB outputs: 64×64, 88×88, 112×112, etc.
                        if tb_out and len(tb_out) == 4:
                            tb_spatial = tb_out[2]  # Height
                            if tb_spatial >= 64:  # Reasonable resolution
                                reason.append(f"✅ TB: {tb_out} (good resolution)")
                                is_good_match = True
                            else:
                                reason.append(f"⚠️  TB: {tb_out} (too low)")
                        
                        # Check FCB output - should be high resolution
                        if fcb_out and len(fcb_out) == 4:
                            fcb_spatial = fcb_out[2]
                            if fcb_spatial >= 352:  # High resolution
                                reason.append(f"✅ FCB: {fcb_out} (high res)")
                            elif fcb_spatial >= 224:
                                reason.append(f"⚠️  FCB: {fcb_out} (medium res)")
                            else:
                                reason.append(f"❌ FCB: {fcb_out} (low res)")
                                is_good_match = False
                    else:
                        reason.append(f"❌ Output: {output.shape}")
                    
                    result['match_quality'] = ' | '.join(reason)
                    result['is_paper_match'] = is_good_match
                    
                    all_results.append(result)
                    
                    if is_good_match:
                        paper_candidates.append(result)
                        print(f"\n🎯 STRONG CANDIDATE FOUND!")
                        print(f"   Model: {model_name}")
                        print(f"   Architecture: {arch_name}")
                        print(f"   Model size: {model_size}")
                        print(f"   TB output: {tb_out}")
                        print(f"   FCB output: {fcb_out}")
                        print(f"   Final output: {output.shape}")
                        
                except Exception as e:
                    # Failed to load with this config
                    continue
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    if paper_candidates:
        print(f"\n✅ Found {len(paper_candidates)} model(s) matching paper specifications!\n")
        
        for candidate in paper_candidates:
            print(f"📁 {candidate['model_name']}")
            print(f"   Architecture: {candidate['architecture']}")
            print(f"   Model size: {candidate['model_size']}")
            print(f"   TB → FCB → Output: {candidate['TB_output']} → {candidate['FCB_output']} → {candidate['final_output']}")
            print(f"   Quality: {candidate['match_quality']}")
            print()
    else:
        print("\n❌ NO MODELS MATCH PAPER SPECIFICATIONS")
        print("\nChecked for:")
        print("  • 512×512 input/output")
        print("  • TB with resolution ≥ 64×64 (not 28×28)")
        print("  • FCB with resolution ≥ 352×352")
        print("\nYou need to retrain with correct architecture.")
    
    print("="*80)
    
    # Save detailed results
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv('detailed_architecture_test.csv', index=False)
    print(f"\n✅ Detailed results saved to: detailed_architecture_test.csv")
    
    return paper_candidates, all_results


# Run it
base_path = "/ediss_data/ediss2/xai-texture/saved_models/FCBFormer"
matches, all_results = comprehensive_model_test(base_path)
