
import onnx
import sys
import types
from unittest.mock import MagicMock
import numpy as np
from onnx import TensorProto

# Monkey Patch for tensorflow_addons (required by onnx-tf but unused for this model)
sys.modules["tensorflow_addons"] = MagicMock()

# Monkey Patch for onnx-tf (expects onnx.mapping which was removed)
if not hasattr(onnx, 'mapping'):
    onnx.mapping = types.ModuleType("mapping")
    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE = {
        TensorProto.FLOAT: np.dtype('float32'),
        TensorProto.UINT8: np.dtype('uint8'),
        TensorProto.INT8: np.dtype('int8'),
        TensorProto.UINT16: np.dtype('uint16'),
        TensorProto.INT16: np.dtype('int16'),
        TensorProto.INT32: np.dtype('int32'),
        TensorProto.INT64: np.dtype('int64'),
        TensorProto.STRING: np.dtype('O'),
        TensorProto.BOOL: np.dtype('bool'),
        TensorProto.FLOAT16: np.dtype('float16'),
        TensorProto.DOUBLE: np.dtype('float64'),
        TensorProto.UINT32: np.dtype('uint32'),
        TensorProto.UINT64: np.dtype('uint64'),
        # Add others if needed
    }
    # Some older onnx-tf might access onnx.mapping.NP_TYPE_TO_TENSOR_TYPE
    onnx.mapping.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.items()}

from onnx_tf.backend import prepare
import tensorflow as tf
import os

def convert(onnx_path, tflite_path):
    print(f"Loading {onnx_path}...")
    onnx_model = onnx.load(onnx_path)
    
    # Check opset
    print(f"Opset version: {onnx_model.opset_import[0].version}")
    
    print("Converting to TensorFlow Representation...")
    # This creates a TensorflowRep object
    tf_rep = prepare(onnx_model)
    
    print("Exporting to SavedModel...")
    saved_model_path = "saved_model_temp"
    tf_rep.export_graph(saved_model_path)
    
    print("Converting SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    # Enable some optimizations for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    print(f"Saving TFLite model to {tflite_path}...")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print("Conversion successful.")

if __name__ == "__main__":
    convert("tb_model_best.onnx", "tb_model_best.tflite")
