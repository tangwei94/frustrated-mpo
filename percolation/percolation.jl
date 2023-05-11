using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2

include("../utils.jl");

L = 6
boundary_condition = :pbc

T = tensor_percolation(0.5, 0.5)
Tdag = mpotensor_dag(T)
𝕋 = mpo_gen(L, T, boundary_condition);
𝕋dag = mpo_gen(L, Tdag, boundary_condition);
𝕋mat = convert_to_mat(𝕋);
𝕋dagmat = convert_to_mat(𝕋dag);

Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
δt = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt.data[end] = 1
ψt = Ut * δt;
ψt

Λt, Ut = eig(𝕋dagmat);
Λt = diag(Λt.data)
δt = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt.data[end] = 1
ψt = Ut * δt;
ψt

Id = id(codomain(𝕋mat));

# find Jordan blocks
for λ in Λt
    rst_t = findall(x->(abs(x-λ) < 1e-6), Λt)
    if length(rst_t) != 1
        S1 = tsvd(𝕋mat - λ * Id)[2].data |> diag
        S2 = tsvd((𝕋mat - λ * Id) * (𝕋mat - λ * Id))[2].data |> diag
        @show λ, length(rst_t)
        @show sum(S1 .> 1e-12), sum(S2 .> 1e-12)
        #@show S1
    end
end

@show Λt
Λm = maximum(norm.(Λt))

segs = (0:100) ./ 100
plot(cos.(2*pi .* segs), sin.(2*pi .* segs), color=:grey, alpha=0.25, linewidth=2, aspect_ratio=:equal, label="")

scatter!(real.(Λt ./ Λm), imag.(Λt ./ Λm), markershape=:x, markersize=8, color=:deepskyblue, alpha=0.8, label="𝕋 spectrum")

xlabel!("Re(λ) / λₘ")
ylabel!("Im(λ) / λₘ")

plot!(size=(400, 400))
