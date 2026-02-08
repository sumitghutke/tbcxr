import sys
import numpy as np
import onnx.helper

# Patch missing function in onnx.helper for onnx_graphsurgeon compatibility
if not hasattr(onnx.helper, 'float32_to_bfloat16'):
    print("Patching onnx.helper.float32_to_bfloat16...")
    def float32_to_bfloat16(x):
        # Dummy implementation: just return something or error if actually called
        # Graphsurgeon only uses it for TYPE MAPPING at import time usually.
        return x 
    onnx.helper.float32_to_bfloat16 = float32_to_bfloat16

from onnx2tf import convert

try:
    convert(
        input_onnx_file_path="tb_model_best.onnx",
        output_folder_path="tflite_model",
        not_use_onnxsim=True, # equivalent to -nos
        non_verbose=True
    )
    print("Conversion Successful via onnx2tf!")
except Exception as e:
    print(f"Conversion failed: {e}")
    sys.exit(1)
