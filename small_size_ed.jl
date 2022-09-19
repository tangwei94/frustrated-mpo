using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

function contraction_info_gen(L::Int)
    ncon_contraction_order = [[ix, -2*ix+1, -2*ix, ix+1] for ix in 1:L] 
    ncon_contraction_order[end][end] = 1
    permutation_orders = Tuple(2 .* (1:L) .- 1), Tuple(2 .* (1:L))
    return ncon_contraction_order, permutation_orders
end

L = 6
boundary_condition = :pbc

ncon_contraction_order, permutation_orders = contraction_info_gen(L);
𝕋 = mpo_gen(L, :frstr, boundary_condition);
𝕋mat = permute(ncon(𝕋.opp, ncon_contraction_order), permutation_orders...);

Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
δt = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt.data[end] = 1
ψt = Ut * δt;

𝔾 = mpo_gen(L, :nonfrstr_adapted, boundary_condition);
𝔾mat = permute(ncon(𝔾.opp, ncon_contraction_order), permutation_orders...);

Λg, Ug = eig(𝔾mat);
Λg = diag(Λg.data)
δg = Tensor(zeros, ComplexF64, ℂ^length(Λg));
δg.data[end] = 1
ψg = Ug * δg;

@show norm(𝔾mat * 𝔾mat' - 𝔾mat' * 𝔾mat) # 𝔾 is normal 
@show norm(𝕋mat * 𝕋mat' - 𝕋mat' * 𝕋mat) # 𝕋 is normal 

@show Λt[end], Λg[end], Λt[end] / Λg[end];
@assert Λt[end] ≈ Λg[end]
Λm = Λt[end]
α = Λt[end] / Λg[end];

Λg .- Λt

segs = (0:100) ./ 100
plot(cos.(2*pi .* segs), sin.(2*pi .* segs), color=:grey, alpha=0.25, linewidth=2, aspect_ratio=:equal, legend=false)
plot!(cos(2*pi/3) .* segs, sin(2*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, legend=false)
plot!(cos(4*pi/3) .* segs, sin(4*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, legend=false)
plot!(cos(0*pi/3) .* segs, sin(0*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, legend=false)
scatter!(real.(Λt ./ Λm), imag.(Λt ./ Λm), color=:deepskyblue, alpha=0.5)
scatter!(real.(Λg ./ Λm), imag.(Λg ./ Λm), color=:red, alpha=0.5)

xlabel!("Re(λ)")
xlabel!("Im(λ)")
