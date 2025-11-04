import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is CPU-only. Install CUDA version:")
    print("pip uninstall torch -y")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu118")
