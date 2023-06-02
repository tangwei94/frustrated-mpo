using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

L = 12

𝕋 = mpo_gen(L, :nonfrstr, :obc);
𝕋t = mpo_gen(L, :nonfrstrT, :obc); 

ψ = FiniteMPS(L, ℂ^4, ℂ^2);

ϕ1 = 𝕋 * 𝕋t * ψ ;
ϕ2 = 𝕋t * 𝕋 * ψ ; 
normalize!(ϕ1);
normalize!(ϕ2);

norm(dot(ϕ1, ϕ2))