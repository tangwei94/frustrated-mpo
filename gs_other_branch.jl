using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choice = :frstr 
boundary_condition = :obc

filename = filename_gen(mpo_choice, boundary_condition)
filename1 = filename * "rot1_"
filename2 = filename * "rot2_"

L = 18

𝕋 = mpo_gen(L, mpo_choice, boundary_condition)

@load filename*"L$(L).jld" ψms
@load filename*"L$(L).jld" vars
@load filename*"L$(L).jld" fs
fs_rot0, ψms_rot0, vars_rot0 = fs, ψms, vars;

@load filename1*"L$(L).jld" ψms
@load filename1*"L$(L).jld" vars
@load filename1*"L$(L).jld" fs
fs_rot1, ψms_rot1, vars_rot1 = fs, ψms, vars;

@load filename2*"L$(L).jld" ψms
@load filename2*"L$(L).jld" vars
@load filename2*"L$(L).jld" fs
fs_rot2, ψms_rot2, vars_rot2 = fs, ψms, vars;

plot()
plot!(abs.(vars_rot0) .+ 1e-16, yaxis=:log)
plot!(abs.(vars_rot1) .+ 1e-16, yaxis=:log)
plot!(abs.(vars_rot2) .+ 1e-16, yaxis=:log)

plot()
plot!(imag.(fs_rot0) * L / (2/3*pi), label="no rotation") 
plot!(imag.(fs_rot1) * L / (2/3*pi), label="rotation 2/3*pi") 
plot!(imag.(fs_rot2) * L / (2/3*pi), label="rotation -2/3*pi") 

plot()
plot!(real.(fs_rot0), label="no rotation")  
plot!(real.(fs_rot1), label="rotation 2/3*pi")  
plot!(real.(fs_rot2), label="rotation -2/3*pi") 
plot!(fill(0.3230659669, length(fs_rot0)), linestyle=:dash, label="exact")

@show dot(ψms_rot1[end], ψms_rot2[end])
@show norm(dot(ψms_rot0[end], ψms_rot2[end]))
@show norm(dot(ψms_rot0[end], ψms_rot1[end]))

@show entanglement_entropy(ψms_rot0[end], L ÷ 2)
@show entanglement_entropy(ψms_rot1[end], L ÷ 2)
@show entanglement_entropy(ψms_rot2[end], L ÷ 2)

nums_rot0 = sample_n_domain_wall(ψms_rot0[end], mpo_choice, boundary_condition)
nums_rot1 = sample_n_domain_wall(ψms_rot1[end], mpo_choice, boundary_condition)
nums_rot2 = sample_n_domain_wall(ψms_rot2[end], mpo_choice, boundary_condition)
plot()
histogram!(nums_rot0, alpha=0.5, color=:red, bins=1:L)
histogram!(nums_rot1, alpha=0.5, color=:green, bins=1:L)
histogram!(nums_rot2, alpha=0.5, color=:blue, bins=1:L)