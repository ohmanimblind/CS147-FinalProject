import torch
import time

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create large random tensors on the GPU
size = 4096
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm-up (important to trigger GPU kernels)
_ = torch.mm(a, b)

# Time the matrix multiplication
start = time.time()
for _ in range(10):
    c = torch.mm(a, b)
torch.cuda.synchronize()  # Wait for GPU to finish
end = time.time()

print("10 matrix multiplications took {:.3f} seconds".format(end - start))

