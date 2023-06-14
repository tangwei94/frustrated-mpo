using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using CairoMakie
using JLD2

include("../utils.jl");

f_exact = 0.3230659669

# construction 1: not C6 symmetric (frustrated MPO)

δ = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2*ℂ^2)
δ[1,1,1,1] = δ[2,2,2,2] = 1

M = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^3*ℂ^3)
M[1, 1, 2] = M[1, 1, 3] = M[1, 2, 1] = M[1, 3, 1] = 1 # spin up 
M[2, 3, 2] = M[2, 2, 3] = M[2, 3, 1] = M[2, 3, 1] = 1 # spin down

@tensor A[-1; -2 -3 -4 -5 -6 -7] := δ[-1; 1 2 3] * M[1; -2 -3] * M[2; -4 -5] * M[3; -6 -7]

# only C3 symmetric 
A1 = permute(A, (1,), (3, 4, 5, 6, 7, 2))
@show norm(A - A1) 
A1 = permute(A, (1,), (4, 5, 6, 7, 2, 3))

# construction 2: C6 symmetric
B = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^3*ℂ^3*ℂ^3*ℂ^3*ℂ^3*ℂ^3) 
for ix in [2, 3], iy in [2, 3], iz in [2, 3]
    B[1, ix, 1, iy, 1, iz, 1] = 1
    B[1, 1, iy, 1, iz, 1, ix] = 1
    B[2, 4-ix, 3, 4-iy, 3, 4-iz, 3] = 1
    B[2, 3, 4-iy, 3, 4-iz, 3, 4-ix] = 1
end
B1 = permute(B, (1,), (3, 4, 5, 6, 7, 2))
@show norm(B - B1)
B1 = permute(B, (1,), (7, 2, 3, 4, 5, 6))
@show norm(B - B1)
B1 = permute(B, (1,), (4, 5, 6, 7, 2, 3))
@show norm(B - B1)

# construction 3: test
T = TensorMap(zeros, ComplexF64, ℂ^3*ℂ^3*ℂ^3, ℂ^3*ℂ^3*ℂ^3) 
for ix in [2, 3], iy in [2, 3], iz in [2, 3]
    T[ix, 1, iy, 1, iz, 1] = 1
    T[1, iy, 1, iz, 1, ix] = 1
    T[4-ix, 3, 4-iy, 3, 4-iz, 3] = 1
    T[3, 4-iy, 3, 4-iz, 3, 4-ix] = 1
end
U, S, V = tsvd(T, (1, 2, 4), (3, 5, 6); trunc=truncerr(1e-14))
@show diag(S.data)
T1 = permute(U * sqrt(S), (1, 2), (3, 4))
T2 = permute(sqrt(S) * V, (1, 2), (3, 4))
δm = isomorphism(ℂ^18, ℂ^3*ℂ^6)
@tensor M1[-1 -2 ; -3 -4] := T1[3 1 ; -3 5] * T2[2 -2 ; 1 4] * δm[-1; 3 2] * δm'[4 5 ; -4]
@tensor M2[-1 -2 ; -3 -4] := T2[3 1 ; -3 5] * T1[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[5 4 ; -4]
𝕋n = DenseMPO([M1, M2])
𝕋n = changebonds(𝕋n, SvdCut(truncerr(1e-10)))

N = 3
𝕋N = reduce(vcat, fill([M1, M2], N));
𝕋N_mat = convert_to_mat(DenseMPO(𝕋N));

@show norm(𝕋N_mat * 𝕋N_mat' - 𝕋N_mat' * 𝕋N_mat)

eigvals(𝕋N_mat.data)


######
𝕋0 = MPOMultiline([T1 T2 ; T2 T1])
χ = 9
ψ0 = MPSMultiline([ℂ^3 ℂ^3; ℂ^3 ℂ^3], [ℂ^χ ℂ^χ; ℂ^χ ℂ^χ])
ψ0, _, _ = leading_boundary(ψ0, 𝕋0, VUMPS(tol_galerkin=1e-10, maxiter=100)); 

# construct PEPS transfer matrix 
function f_PEPS_transfer_mat(A::AbstractTensorMap, χ::Int)
    δ1 = isomorphism(ℂ^(χ^2), (ℂ^χ)'*ℂ^χ)
    δ1 = permute(δ1, (1, 2), (3, ))

    δ2 = isomorphism((ℂ^(χ^2))', (ℂ^χ)'*ℂ^χ)
    δ2 = permute(δ2, (2, ), (3, 1))

    @tensor T[-1 -2 -3; -4 -5 -6] := A[1; 4 2 8 10 12 6] * A'[5 3 9 11 13 7; 1] * δ1[-1 2; 3] * δ1[-2 4; 5] * δ1[-3 6; 7] * δ2[8; 9 -4] * δ2[10; 11 -5] * δ2[12; 13 -6];

    U, S, V = tsvd(T, (1, 2, 4), (3, 5, 6); trunc=truncdim(χ^2))
    @show diag(S.data)
    T1 = permute(U * sqrt(S), (1, 2), (3, 4))
    T2 = permute(sqrt(S) * V, (1, 2), (3, 4))

    δm = isomorphism(ℂ^(χ^4), ℂ^(χ^2)*ℂ^(χ^2))
    @tensor M1[-1 -2 ; -3 -4] := T1[3 1 ; -3 5] * T2[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]
    @tensor M2[-1 -2 ; -3 -4] := T2[3 1 ; -3 5] * T1[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]

    𝕋 = MPOMultiline([T1 T2 ; T2 T1])
    𝕄 = DenseMPO([M1, M2])
    ℕ = DenseMPO([T1, T2])
    return 𝕋, 𝕄, ℕ 
end

𝕋1, 𝕄1, ℕ1 = f_PEPS_transfer_mat(A, 3);
𝕋2, 𝕄2, ℕ2 = f_PEPS_transfer_mat(B, 3);

χ = 9
ψ0 = MPSMultiline([ℂ^9 ℂ^9; ℂ^9 ℂ^9], [ℂ^χ ℂ^χ; ℂ^χ ℂ^χ])
expand_alg = OptimalExpand(truncdim(9))

ψ1, envs1, _ = leading_boundary(ψ0, 𝕋1, VUMPS(tol_galerkin=1e-10, maxiter=100)); 
ψ2, envs2, _ = leading_boundary(ψ0, 𝕋2, VUMPS(tol_galerkin=1e-10, maxiter=100)); 

ψ0 = InfiniteMPS([ℂ^9, ℂ^9], [ℂ^χ, ℂ^χ]);
ψb, envsb, _ = leading_boundary(ψ0, 𝕄1, VUMPS(tol_galerkin=1e-12, maxiter=100)); 
ψb, envsb, _ = leading_boundary(ψ0, 𝕄2, VUMPS(tol_galerkin=1e-12, maxiter=100));

ψ1 = ψ0
alg = SvdCut(truncdim(9))
for ix in 1:10
    ψ1 = changebonds(ℕ1 * ψ1, alg)
    ψ2 = InfiniteMPS([ψ1.AL[2], ψ1.AL[1]])
    @show ix, left_virtualspace(ψ1, 1), dot(ψ2, ℕ1, ψ1) / dot(ψ2, ψ1), entropy(ψ1, 1)
    ψ1 = ψ2
end
ψ1 = ψ0
for ix in 1:50
    ψ1 = changebonds(ℕ2 * ψ1, alg)
    ψ2 = InfiniteMPS([ψ1.AL[2], ψ1.AL[1]])
    @show ix, left_virtualspace(ψ1, 1), log(norm(dot(ψ1, ℕ2, ψ1) / dot(ψ2, ψ1))), real(entropy(ψ1, 1)), real(entropy(ψ2, 1))
    ψ1 = ψ2
end

ψ1 = ψ0
for ix in 1:50
    ψ1 = changebonds(𝕄2 * ψ1, alg)
    ψ2 = InfiniteMPS([ψ1.AL[2], ψ1.AL[1]])
    @show ix, left_virtualspace(ψ1, 1), dot(ψ2, 𝕄2, ψ1) / dot(ψ2, ψ1), real(entropy(ψ1, 1)), real(entropy(ψ2, 1))
    ψ1 = ψ2
end