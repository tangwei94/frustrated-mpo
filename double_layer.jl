using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

χs = [2, 4, 8, 16, 32, 64]

𝕋 = mpo_gen(1, :frstr, :inf)

f, vars, diffs, ψms = power_projection(𝕋*𝕋, χs; operation=no_operation, filename = "doublelayer")