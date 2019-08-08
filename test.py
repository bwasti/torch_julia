import torch
import torch_julia

torch_julia.load("test.jl")

a = torch.randn(100,100)
b = torch.randn(100,100)

c = torch.ops.julia.abs_sqrt(b)
torch.testing.assert_allclose(c, torch.sqrt(torch.abs(b)))

print("trying mm")
torch.testing.assert_allclose(torch.mm(a,b), torch.ops.julia.mm(a,b), atol=0.001, rtol=0.001)

import time

t = time.time()
for i in range(10):
  _ = torch.mm(a,b)
print("torch", time.time() - t)
t = time.time()
for i in range(10):
  _ = torch.ops.julia.mm(a,b)
print("julia", time.time() - t)
print("passed.")
