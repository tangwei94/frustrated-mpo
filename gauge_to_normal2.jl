using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using CairoMakie

include("utils.jl");

A = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2);
X = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2);
Adata = reshape(A.data, (2,2,2,2))
Adata = Adata + conj.(permutedims(Adata,(1, 3, 2, 4)));
Adata = reshape(Adata, (4,4))
A = TensorMap(Adata, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

X = add_util_leg(X)
𝔸 = DenseMPO([A])
𝕏 = DenseMPO([X])

𝕋 = 𝔸 * 𝕏

expand_alg = OptimalExpand(truncdim(2))

for ix in 1:3
    ψ0 = InfiniteMPS([ℂ^2], [ℂ^(2*ix)])
    ψR, envR, _ = leading_boundary(ψ0, 𝕋, VUMPS(tol_galerkin=1e-12, maxiter=100))
    ψL = 𝕏 * ψR
    @show norm(dot(ψL, 𝕋, ψR)) / norm(dot(ψL, ψR))
    @show dot(ψL, ψR)
    @show entropy(ψR)
end

f1, vars1, diffs1, ψms1 = power_projection(𝕋, [1, 2, 4, 6, 8]; Npower=100, filename="tmpaaa");

for ψR in ψms1 
    ψL = 𝕏 * ψR
    @show norm(dot(ψL, 𝕋, ψR)) / norm(dot(ψL, ψR))
    @show dot(ψL, 𝕋, ψR) / dot(ψL, ψR)
    @show entropy(ψR)
end

