import ai_edge_torch
import torch
import numpy as np
import tensorflow as tf
import os

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    if size < 1024:
        return f"{size} Bytes"
    elif size < 1024**2:
        return f"{size/1024:.2f} KB"
    else:
        return f"{size/1024**2:.2f} MB"

model_path = 'final_pruned_model.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

original_params = count_model_parameters(model)
print(f"Original PyTorch Model Parameters: {original_params:,}")
print(f"Original PyTorch Model Size: {get_file_size(model_path)}")

device = torch.device("cpu")
model.to(device)

sample_inputs = (torch.randn(1, 3, 224, 224).to(device),)

torch_output = model(*sample_inputs)

edge_model = ai_edge_torch.convert(model, sample_inputs)

edge_output = edge_model(*sample_inputs)

if np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5, rtol=1e-5):
    print("Inference result with PyTorch and LiteRT was within tolerance.")
else:
    print("Warning: PyTorch -> LiteRT conversion may have issues.")

print("Starting TensorFlow Lite Quantization...")

tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}

tfl_drq_model = ai_edge_torch.convert(
    model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
)

output_path = 'final_pruned_model_tflite_quant.tflite'
tfl_drq_model.export(output_path)

print(f"TFLite Quantized Model exported as '{output_path}'.")
print(f"Quantized TFLite Model Size: {get_file_size(output_path)}")