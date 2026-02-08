#!/bin/bash
# convert_onnx_to_tflite.sh

echo "Converting tb_model_best.onnx to TFLite using onnx2tf..."

# Run onnx2tf
# -i: Input ONNX file
# -o: Output directory (onnx2tf creates a directory)
# -osd: Output separator for debug (optional)
# -co: Compress op (optional, usually good for mobile) -> ignoring for now unless needed
onnx2tf -i tb_model_best.onnx -o tflite_model

echo "Conversion complete. Checking output..."
ls -l tflite_model/
