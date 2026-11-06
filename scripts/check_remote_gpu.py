import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("Remote GPU is unavailable")

gpu = torch.cuda.get_device_properties(0)

print("GPU:", gpu.name)
print("GPU memory:", round(gpu.total_memory / (1024 ** 3), 2), "GB")

x = torch.randn(3000, 3000, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU computation: SUCCESS")
print("Output shape:", tuple(y.shape))
