using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using CairoMakie
using JLD2

include("../utils.jl");

A = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^3*ℂ^3*ℂ^3*ℂ^3*ℂ^3*ℂ^3) 
A[1, 1, 3, 3, 3, 3, 3] = A[1, 3, 1, 3, 3, 3, 3] = A[1, 3, 3, 1, 3, 3, 3] = A[1, 3, 3, 3, 1, 3, 3] = A[1, 3, 3, 3, 3, 1, 3] = A[1, 3, 3, 3, 3, 3, 1] = 1
A[2, 2, 3, 3, 3, 3, 3] = A[2, 3, 2, 3, 3, 3, 3] = A[2, 3, 3, 2, 3, 3, 3] = A[2, 3, 3, 3, 2, 3, 3] = A[2, 3, 3, 3, 3, 2, 3] = A[2, 3, 3, 3, 3, 3, 2] = 1

B = TensorMap(ComplexF64[0 1 0 ; -1 0 0 ; 0 0 1], ℂ^3, ℂ^3)

@tensor Afull[-1; -2 -3 -4 -5 -6 -7] := A[-1; 1 2 3 -5 -6 -7] * B[1; -2] * B[2; -3] * B[3; -4]; 

function f_PEPS_transfer_mat(A::AbstractTensorMap, χ::Int)
    δ1 = isomorphism(ℂ^(χ^2), (ℂ^χ)'*ℂ^χ)
    δ1 = permute(δ1, (1, 2), (3, ))

    δ2 = isomorphism((ℂ^(χ^2))', (ℂ^χ)'*ℂ^χ)
    δ2 = permute(δ2, (2, ), (3, 1))

    @tensor T[-1 -2 -3; -4 -5 -6] := A[1; 4 2 8 10 12 6] * A'[5 3 9 11 13 7; 1] * δ1[-1 2; 3] * δ1[-2 4; 5] * δ1[-3 6; 7] * δ2[8; 9 -4] * δ2[10; 11 -5] * δ2[12; 13 -6];

    U, S, V = tsvd(T, (1, 2, 4), (3, 5, 6); trunc=truncerr(1e-10))
    @show length(diag(S.data)), S.data[end, end]
    T1 = permute(U * sqrt(S), (1, 2), (3, 4))
    T2 = permute(sqrt(S) * V, (1, 2), (3, 4))

    #δm = isomorphism(ℂ^(χ^4), ℂ^(χ^2)*ℂ^(χ^2))
    #@tensor M1[-1 -2 ; -3 -4] := T1[3 1 ; -3 5] * T2[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]
    #@tensor M2[-1 -2 ; -3 -4] := T2[3 1 ; -3 5] * T1[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]

    ℕ = DenseMPO([T1, T2])
    ℕ1 = changebonds(ℕ, SvdCut(truncerr(1e-10)))
    T1, T2 = ℕ1.opp[1], ℕ1.opp[2]

    @show MPSKit._firstspace(T1), MPSKit._firstspace(T2)
    𝕋 = MPOMultiline([T1 T2 ; T2 T1])
    return 𝕋, T1, T2 
end

𝕋, T1, T2 = f_PEPS_transfer_mat(Afull, 3);

norm(mpotensor_dag(T1) - T1)
norm(mpotensor_dag(T2) - T2)

ϕ1 = convert(InfiniteMPS, 𝕋.data[1] * 𝕋.data[2])
ϕ2 = convert(InfiniteMPS, 𝕋.data[2] * 𝕋.data[1])
dot(ϕ1, ϕ2)

# VUMPS 
ψ = MPSMultiline([ℂ^9 ℂ^9; ℂ^9 ℂ^9], [ℂ^χ ℂ^χ; ℂ^χ ℂ^χ])
ψ1, _, _ = leading_boundary(ψ, 𝕋, VUMPS(tol_galerkin=1e-10, maxiter=1000));



