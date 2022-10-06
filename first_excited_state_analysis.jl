using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choiceR = :frstr # :nonfrstr, :frstrT
mpo_choiceL = :frstrT # :nonfrstr, :frstrT
boundary_condition = :obc # :pbc, :obc

filenameR = filename_gen(mpo_choiceR, boundary_condition)
filenameL = filename_gen(mpo_choiceL, boundary_condition)

ovlps1L = Float64[]
ovlps1R = Float64[]

Ls = [6, 12, 18, 24, 30, 36, 48, 60]; 

plot()
for L in Ls
    @load filenameR * "L$(L).jld" ψms
    ψR = copy(ψms[end]);
    @load filenameL * "L$(L).jld" ψms
    ψL = copy(ψms[end]);

    @load filenameR * "1st_excitation_L$(L).jld" ψms
    ψ1R = copy(ψms[end]);
    @load filenameL * "1st_excitation_L$(L).jld" ψms
    ψ1L = copy(ψms[end]);

    𝕋R = mpo_gen(L, mpo_choiceR, boundary_condition);
    𝕋L = mpo_gen(L, mpo_choiceL, boundary_condition);

    @show norm(ψR), norm(ψL), norm(ψ1R), norm(ψ1L)

    push!(ovlps1R, norm(dot(ψ1R, ψR)))
    push!(ovlps1L, norm(dot(ψ1L, ψL)))
end
scatter!(Ls, 1 .- ovlps1R, yaxis=:log, label="1 - <ψR1|ψR0>")

xlabel!("L")
ylabel!("1 - <ψ1|ψ0>")