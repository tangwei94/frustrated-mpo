using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

global mpo_choice, boundary_condition, filename, operation

if length(ARGS) == 2
    mpo_choice = Symbol(ARGS[1])
    boundary_condition = Symbol(ARGS[2])
    operation = gs_operation
    filename = filename_gen(mpo_choice, boundary_condition)
elseif length(ARGS) == 3
    mpo_choice = Symbol(ARGS[1])
    boundary_condition = Symbol(ARGS[2])
    rotation = Symbol(ARGS[3])
    if rotation == :rot1 
        operation = gs1_operation
    else rotation = :rot2 
        operation = gs2_operation
    end
    filename = filename_gen(mpo_choice, boundary_condition; more_info=ARGS[3]*"_")
else
    mpo_choice = :frstr # :nonfrstr, :frstrT, :nonfrstrT
    boundary_condition = :obc # :pbc, :obc
    operation = gs_operation
    filename = filename_gen(mpo_choice, boundary_condition)
end

@show filename

Ls = [6, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96]; 
χs = [4, 8, 12, 16, 20, 24, 28, 32];
for L in Ls 
    𝕋 = mpo_gen(L, mpo_choice, boundary_condition);
    fs, vars, diffs, ψms = power_projection(𝕋, χs; filename=filename, operation=operation);
end