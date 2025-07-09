import torch

print("Hello from the GPU server!")
print("PyTorch version:", torch.__version__)
print("CUDA available?", torch.cuda.is_available())