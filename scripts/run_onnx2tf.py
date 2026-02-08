
import sys
import onnx.helper
import numpy as np

# Monkey patch for onnx <-> onnx-graphsurgeon compatibility
if not hasattr(onnx.helper, 'float32_to_bfloat16'):
    print("Patching onnx.helper.float32_to_bfloat16...")
    def float32_to_bfloat16(x):
        # minimal mock, just cast to uint16 if needed, or return generic
        # This is likely only used for constant folding/export of BF16 which we don't present
        return np.float32(x).view(np.uint16)[1::2] 
    onnx.helper.float32_to_bfloat16 = float32_to_bfloat16

from onnx2tf import main

if __name__ == '__main__':
    sys.exit(main())
