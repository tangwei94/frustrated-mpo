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

ϕ0 = circular_mps(ψ, L, 0);
ϕp = circular_mps(ψ, L, 0.2); 
ϕm = circular_mps(ψ, L, -0.2); 

Uy = DenseMPO(fill(add_util_leg(exp(-im*pi*σy/4)), L))
function meas_σx(ψlp)

    σxs = perfect_sampling(Uy * ψlp)

    return sum(2 .* σxs .- 3)

end

X0s = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_σx(ϕ0)
end
Xps = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_σx(ϕp)
end
Xms = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_σx(ϕm)
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 900))
ax1 = Axis(fig[1, 1], xlabel=L"X", ylabel=L"N")
hist!(ax1, X0s; bins=-L:L, label=L"\tau=0")
axislegend(ax1; position=:rt)
@show fig

ax2 = Axis(fig[2, 1], xlabel=L"X", ylabel=L"N")
hist!(ax2, Xps; bins=-L:L, label=L"\tau=0.2")
axislegend(ax2; position=:rt)
@show fig

ax3 = Axis(fig[3, 1], xlabel=L"X", ylabel=L"N")
hist!(ax3, Xms; bins=-L:L, label=L"\tau=-0.2")
axislegend(ax3; position=:rt)
@show fig

save("square_ising/data/VUMPS_check_distribution.pdf", fig)