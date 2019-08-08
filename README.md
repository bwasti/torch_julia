# `torch_julia`

This project makes it easy to write operations (or really anything) in Julia and use those within PyTorch.

# Build

```
./build.sh
```

# Usage

In a Julia file, you'll need to use `TorchReg.torch_reg(f)` to expose a function to PyTorch.

```
function f(a, b)
  c = a .* b
  return c
end

import Pkg; Pkg.activate(".") # or install TorchReg
using TorchReg
torch_reg(f)
```

In a Python file, you can then use `torch_julia.load("julia_code.jl")` to expose the functions registered
to the `torch.ops.julia._` namespace.

```
import torch
import torch_julia
torch_julia.load("julia_code.jl")

a = torch.randn(10)
b = torch.randn(10)
print(a,b)
c = torch.ops.julia.f(a, b)
print(c)
```

# Gotchas

- This is basically a prototype
- Only 1D and 2D arrays are supported at the moment
- The JIT backend isn't used for composed Julia calls (it should be)
- You'll probably need to `LD_PRELOAD` the Julia library (see https://discourse.julialang.org/t/calling-julia-shared-library-from-python-linking-libraries/9169)

# Test

```
LD_PRELOAD=/data/users/bwasti/julia/usr/lib/libjulia.so PYTHONPATH=build python test.py
```

