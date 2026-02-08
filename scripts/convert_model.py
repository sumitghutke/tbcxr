
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Define the model class exactly as in model.py to load weights
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        # Initialize with same structure
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        return self.model(x)

def convert_to_onnx(model_path, output_path):
    device = torch.device('cpu')
    model = DenseNet121()
    
    print(f"Loading model from {model_path}...")
    try:
        # Load state dict
        # Note: In model.py, we save self.model.state_dict(), so keys start with 'features...' 
        # But our wrapper class has 'model.features...'. 
        # The verify_accuracy fix used model.model.load_state_dict().
        # Let's verify how it's saved.
        # model.py: self.model.state_dict() -> this means keys are 'features.conv0...', 'classifier.0.weight' etc.
        # Our DenseNet121 wrapper has self.model.
        # So we should load into model.model.
        
        state_dict = torch.load(model_path, map_location=device)
        model.model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Dummy input for tracing
    # Shape: (Batch, Channels, Height, Width) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_in = "tb_model_best.pt"
    else:
        model_in = sys.argv[1]
        
    onnx_out = model_in.replace(".pt", ".onnx")
    convert_to_onnx(model_in, onnx_out)
