using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs

function circular_mps(ψ, L, τ, ϵ=1e-8)
    G = exp(-τ * σx)
    @tensor A[-1 -2 ; -3] := ψ.AL[1][-1 1; -3] * G[-2 ; 1]
    return circular_mps(A, L, ϵ)
end

L = 16
ψ = ψs[4]
τ = 0.

ψlp = circular_mps(ψ, L, τ);

Uy = DenseMPO(fill(add_util_leg(exp(-im*pi*σy/4)), L))
function meas_σx()

    σxs = perfect_sampling(Uy * ψlp)

    return sum(2 .* σxs .- 3)

end

Xs = map(1:100*L) do ix
    @show ix 
    return meas_σx()
end

bins = findmax(Xs)[1] - findmin(Xs)[1]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"X", ylabel=L"N")
xlims!(ax1, (-L, L))
hist!(ax1, Xs; bins=bins)
@show fig