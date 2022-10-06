using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choiceR = :frstr # :nonfrstr, :frstrT
mpo_choiceL = :frstrT # :nonfrstr, :frstrT
boundary_condition = :obc # :pbc, :obc

filenameR = filename_gen(mpo_choiceR, boundary_condition)
filenameL = filename_gen(mpo_choiceL, boundary_condition)

Ls = [6, 12, 18, 24, 30]; 
#Ls = [72, 84, 96]; 
χs = [4, 8, 12, 16, 20, 24, 28, 32]; 

for L in Ls
    @load filenameR * "L$(L).jld" ψms
    ψR = copy(ψms[end]);
    @load filenameL * "L$(L).jld" ψms
    ψL = copy(ψms[end]);

    𝕋R = mpo_gen(L, mpo_choiceR, boundary_condition);
    𝕋L = mpo_gen(L, mpo_choiceL, boundary_condition);

    obtain_1st_excitation_R = operation_scheme(0.5, 0, [ψL], [ψR]);
    obtain_1st_excitation_L = operation_scheme(0.5, 0, [ψR], [ψL]);

    fsR, varsR, diffsR, ψmsR = power_projection(𝕋R, χs; operation=obtain_1st_excitation_R, filename=filenameR * "1st_excitation_"); 
    fsL, varsL, diffsL, ψmsL = power_projection(𝕋L, χs; operation=obtain_1st_excitation_L, filename=filenameL * "1st_excitation_"); 

    plot()
    plot!(varsR, yaxis=:log)
    plot!(varsL, yaxis=:log)

    ψ1L, ψ1R = copy(ψmsL[end]), copy(ψmsR[end]);

    @show norm(dot(ψ1R, ψR))
    @show norm(dot(ψ1L, ψL))
    @show norm(dot(ψ1L, ψ1R))

    @show log(dot(ψL, 𝕋R, ψR) / dot(ψL, ψR)) / L
    @show log(dot(ψ1L, 𝕋R, ψ1R) / dot(ψ1L, ψ1R)) / L
end

#obtain_2nd_excitation_R = operation_scheme(2, 0.005*pi, [ψL, ψ1L], [ψR, ψ1R]);
#obtain_2nd_excitation_L = operation_scheme(2, 0.005*pi, [ψR, ψ1R], [ψL, ψ1L]);
#
#fsR, varsR, diffsR, ψmsR = power_projection(𝕋R, χs; operation=obtain_2nd_excitation_R); 
#fsL, varsL, diffsL, ψmsL = power_projection(𝕋L, χs; operation=obtain_2nd_excitation_L); 
#
#ψ2L, ψ2R = copy(ψmsL[end]), copy(ψmsR[end]);
#
#plot()
#plot!(varsR, yaxis=:log)
#plot!(varsL, yaxis=:log)
#
#norm(dot(ψ2R, ψR))
#norm(dot(ψ2L, ψL))
#norm(dot(ψ2L, ψ2R))
#
#log(dot(ψL, 𝕋R, ψR) / dot(ψL, ψR)) / L
#log(dot(ψ1L, 𝕋R, ψ1R) / dot(ψ1L, ψ1R)) / L
#log(dot(ψ2L, 𝕋R, ψ2R) / dot(ψ2L, ψ2R)) / L 
#angle(dot(ψ2L, 𝕋R, ψ2R) / dot(ψ2L, ψ2R)) / L