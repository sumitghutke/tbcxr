import torch
import model
import os

def inspect_checkpoint(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Inspecting {filepath}...")
    try:
        # Load to CPU to avoid CUDA errors if GPU not available
        state_dict = torch.load(filepath, map_location=torch.device('cpu'))
        print(f"Loaded state_dict with {len(state_dict)} keys.")
        print("Sample keys from checkpoint:")
        for i, key in enumerate(state_dict.keys()):
            if i >= 5: break
            print(f" - {key}")
            
        # Initialize model
        net = model.DenseNet121()
        model_keys = net.model.state_dict().keys()
        print(f"\nCurrent DenseNet121.model has {len(model_keys)} keys.")
        print("Sample keys from current model:")
        for i, key in enumerate(model_keys):
            if i >= 5: break
            print(f" - {key}")
            
        # Check for strict match
        missing = set(model_keys) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(model_keys)
        
        print(f"\nMissing keys (in model but not checkpoint): {len(missing)}")
        if missing:
            print(f"Sample missing: {list(missing)[:3]}")
            
        print(f"Unexpected keys (in checkpoint but not model): {len(unexpected)}")
        if unexpected:
            print(f"Sample unexpected: {list(unexpected)[:3]}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint("tb_model_best.pt")
