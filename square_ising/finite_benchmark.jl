using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");

βc = asinh(1) / 2

T = tensor_square_ising(βc)

L = 6

𝕋 = mpo_gen(L, T, :pbc);
𝕋mat = convert_to_mat(𝕋);

Λ, U = eigh(𝕋mat)
f = - log(Λ.data[end, end]) / βc / L