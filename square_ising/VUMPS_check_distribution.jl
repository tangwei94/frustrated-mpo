using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)

@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs

function circular_mps(ψ, L, G, ϵ=1e-8)
    @tensor A[-1 -2 ; -3] := ψ.AL[1][-1 1; -3] * G[-2 ; 1]
    sp = left_virtualspace(ψ, 1)
    B = permute(isomorphism(ℂ^1*sp', sp), (1,), (2, 3,))
    t = id(sp)
    M = B
    # As = typeof(A)[]
    # for ix in 1:L
    #     push!(As, L1)
    # end
    As = map(1:L) do ix
        if ix < L
            @tensor A1[-1 -2 ; -3 -4] := M[-1 ; 1 2] * A[1 -2 ; -3] * t[2; -4]
            L1, M = leftorth(A1, (1, 2), (3, 4))
        else
            @tensor A1[-1 -2 ; -3] := M[-1; 1 2] * A[1 -2; 3] * t[2; 4] * B'[3 4 ; -3]
            L1, M = leftorth(A1, (1, 2), (3,))
        end
        return L1
    end

    M = id(ℂ^1)
    ix = 1
    for ix in L:-1:1
        A1 = As[ix] * M
        M, S, R1 = tsvd(A1, (1,), (2, 3), trunc=truncerr(ϵ))
        As[ix] = permute(R1, (1, 2), (3,))
        M = M * S
    end

    return FiniteMPS(As)
end

L = 16
ψ = ψs[4]
τ = 0.
G = exp(-τ * σx)

ψlp = circular_mps(ψ, L, G);

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