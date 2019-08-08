function abs_sqrt(x)
  return map(k -> map(sqrt, map(abs, k)), x)
end

using TensorOperations

function mm(X, Y)
  @tensor begin
    E[i, k] := X[i, j] * Y[j, k]
  end
  E
end

import Pkg; Pkg.activate(".")
using TorchReg 
torch_reg(abs_sqrt)
torch_reg(mm)
