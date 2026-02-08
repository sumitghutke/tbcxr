
import tensorflow as tf
import numpy as np
import os

def check_tflite_model(model_path):
    print(f"Checking {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} does not exist.")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        
        # Test inference with dummy data
        input_shape = input_details[0]['shape']
        print(f"Input shape: {input_shape}")
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Output shape: {output_data.shape}")
        print(f"Output data sample: {output_data}")
        print("TFLite model verification successful.")
        
    except Exception as e:
        print(f"Error checking TFLite model: {e}")

if __name__ == "__main__":
    # onnx2tf usually outputs model as saved_model and .tflite in the output dir
    # Based on our script: -o tflite_model
    # The default name is usually 'model_float32.tflite' or 'tb_model_best_float32.tflite'
    
    # We will search for .tflite files in the directory
    model_dir = "tflite_model"
    if os.path.exists(model_dir):
        files = [f for f in os.listdir(model_dir) if f.endswith('.tflite')]
        if files:
            for f in files:
                check_tflite_model(os.path.join(model_dir, f))
        else:
            print("No .tflite files found in tflite_model/")
    else:
        print("tflite_model/ directory not found.")
