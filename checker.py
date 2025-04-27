import torch
import subprocess

# PyTorch GPU information
print("===== PyTorch GPU Information =====")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
else:
    print("No GPU available according to PyTorch")

# NVIDIA System Management Interface (nvidia-smi) output
print("\n===== NVIDIA-SMI Output =====")
try:
    nvidia_smi_output = subprocess.check_output(['nvidia-smi'],
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
    print(nvidia_smi_output)
except (subprocess.SubprocessError, FileNotFoundError):
    print("nvidia-smi command not found or failed to execute")