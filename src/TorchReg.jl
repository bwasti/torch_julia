module TorchReg 

torch_funcs = []

function params(f)::Array{Int64}
  p = []
  for n in methods(f)
    push!(p, length(n.sig.parameters)-1)
  end
  p
end

struct Sig
  name::Symbol
  params::Array{Int64}
end

function torch_reg(f)
  t::Sig = Sig(Symbol(f), params(f))
  push!(torch_funcs, t)
end

export torch_reg
export torch_funcs

end
