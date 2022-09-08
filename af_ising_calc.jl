using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

if length(ARGS) == 2
    mpo_choice = Symbol(ARGS[1])
    boundary_condition = Symbol(ARGS[2])
else
    mpo_choice = :frstr # :nonfrstr, :frstrT
    boundary_condition = :obc # :pbc, :obc
end

filename = filename_gen(mpo_choice, boundary_condition)
@show filename

Ls = [6, 12, 18, 24, 30, 36, 48, 60]; 
χs = [4, 8, 12, 16, 20, 24, 28, 32];
for L in Ls 
    𝕋 = mpo_gen(L, mpo_choice, boundary_condition);
    fs, vars, diffs, ψms = power_projection(𝕋, χs; filename=filename);
end
