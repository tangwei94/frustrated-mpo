using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using CairoMakie
using JLD2

include("../utils.jl");

A = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^3*ℂ^3*ℂ^3*ℂ^3) 
A[1, 1, 3, 3, 3] = A[1, 3, 1, 3, 3] = A[1, 3, 3, 1, 3] = A[1, 3, 3, 3, 1] = 1
A[2, 2, 3, 3, 3] = A[2, 3, 2, 3, 3] = A[2, 3, 3, 2, 3] = A[2, 3, 3, 3, 2] = 2

B = TensorMap(ComplexF64[0 1 0 ; -1 0 0 ; 0 0 1], ℂ^3, ℂ^3)

@tensor Afull[-1; -2 -3 -4 -5] := A[-1; -2 -3 1 2] * B[1; -4] * B[2; -5]

χ = 3
δ1 = isomorphism(ℂ^(χ^2), (ℂ^χ)'*ℂ^χ)
δ1 = permute(δ1, (1, 2), (3, ))

δ2 = isomorphism((ℂ^(χ^2))', (ℂ^χ)'*ℂ^χ)
δ2 = permute(δ2, (2, ), (3, 1))

@tensor T[-1 -2; -3 -4] := Afull[9; 4 2 6 8] * Afull'[3 1 5 7; 9] * δ1[-1 2 ; 1] * δ1[-2 4; 3] * δ2[6; 5 -3] * δ2[8; 7 -4]

𝕋 = DenseMPO([T])

𝕋1 = changebonds(𝕋, SvdCut(truncerr(1e-14)));

@show left_virtualspace(convert(InfiniteMPS, 𝕋1), 1)

T1dag = mpotensor_dag(𝕋1.opp[1]);
𝕋1dag = DenseMPO([T1dag]);

# check normality 
ϕ1 = convert(InfiniteMPS, 𝕋1 * 𝕋1dag);
ϕ2 = convert(InfiniteMPS, 𝕋1dag * 𝕋1);
@show dot(ϕ1, ϕ2)

# VUMPS 
ψ = InfiniteMPS([ℂ^9], [ℂ^18]);
ψ1, env1, _ = leading_boundary(ψ, 𝕋1, VUMPS(tol_galerkin=1e-10, maxiter=1000));
expand_alg = OptimalExpand(truncdim(6))
ψ2, _ = changebonds(ψ1, 𝕋1, expand_alg, env1)

dot(ψ1, 𝕋1, ψ1)

ψ2, env2, _ = leading_boundary(ψ2, 𝕋1, VUMPS(tol_galerkin=1e-10, maxiter=1000))

