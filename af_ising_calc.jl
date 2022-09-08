# use power method to find the fix point of the transfer matrix
# for the frustrated MPO, have to use the "constant shift" trick

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

if mpo_choice == :frstr
    d_ph = 2;
    filename = "frustrated_";
elseif mpo_choice == :nonfrstr 
    d_ph = 4;
    filename = "nonfrustrated_";
elseif mpo_choice == :frstrT 
    d_ph = 2;
    filename = "frustrated_T_";
end
if boundary_condition == :obc 
    filename = filename * "obc_"
else
    filename = filename * "pbc_"
end

@show filename

#Ls = [6, 12, 18, 24, 30, 36, 48, 60]; 
#χs = [4, 8, 12, 16, 20, 24, 28, 32];
Ls = [6, 12]; 
χs = [4, 8];
for L in Ls 
    𝕋 = mpo_gen(Ls[end], mpo_choice, boundary_condition);
    fs, vars, diffs, ψms = power_projection(𝕋, χs; filename=filename);
end
