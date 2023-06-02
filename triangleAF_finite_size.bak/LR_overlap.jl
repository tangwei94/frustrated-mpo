using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choiceR = :frstr # :nonfrstr, :frstrT
mpo_choiceL = :frstrT # :nonfrstr, :frstrT
boundary_condition = :obc # :pbc, :obc

filenameR = filename_gen(mpo_choiceR, boundary_condition)
filenameL = filename_gen(mpo_choiceL, boundary_condition)

Ls = [6, 12, 18, 24, 30, 36, 48, 60, 72, 84]; 
χs = [4, 8, 12, 16, 20, 24, 28, 32];

ψmsR_L = []
ψmsL_L = []
for L in Ls
    @load filenameR*"L$(L).jld" ψms
    push!(ψmsR_L, ψms)
    @load filenameL*"L$(L).jld" ψms
    push!(ψmsL_L, ψms)
end

# overlaps
ovlps = [dot(ψmsL[end], ψmsR[end]) for (ψmsL, ψmsR) in zip(ψmsL_L, ψmsR_L)] 
scatter(Ls, norm.(ovlps), yaxis=:log, legend=false)
xlabel!("L")
ylabel!("norm(<L|R>)")
savefig(filenameR*"LRoverlap.pdf")

# domain wall distribution
σsR = sample_n_domain_wall(ψmsR_L[end][end], mpo_choiceR, boundary_condition)
σsL = sample_n_domain_wall(ψmsL_L[end][end], mpo_choiceL, boundary_condition)

plot()
xlabel!("number of domain walls")
ylabel!("number of samples (1000 in total)")
histogram!(σsR, alpha=0.5, color=:red, bins=1:Ls[end], label="|R>")
histogram!(σsL, alpha=0.5, color=:blue, bins=1:Ls[end], label="|L>")
plot!(fill(2/3*Ls[end], 2), [0, 150], color=:grey, alpha=0.5, linestyle=:dash, linewidth=2, label="2/3*L")
savefig(filenameR*"domainwall_cfgs.pdf")

# free energy 
𝕋s = [mpo_gen(L, mpo_choiceR, boundary_condition) for L in Ls]; 
fs = [log(dot(ψmsL[end], 𝕋, ψmsR[end]) / dot(ψmsL[end], ψmsR[end])) / length(𝕋) for (ψmsL, 𝕋, ψmsR) in zip(ψmsL_L, 𝕋s, ψmsR_L)] 

scatter(1 ./ Ls, real.(fs), legend=false)
plot!([0; 1 ./ Ls], fill(exact_free_energy, length(Ls) + 1))

M = fill(1.0, (length(Ls[4:end]), 2))
M[:, 2] = 1 ./ Ls[4:end]
fits = (M' * M) \ (M' * fs[4:end]) # linear fitting

@show fits[1]|>real, exact_free_energy
(fits[1] - exact_free_energy) / exact_free_energy
plot!([0; 1 ./ Ls], real.(fits[1] .+ fits[2] .* [0; 1 ./ Ls])) 

xlabel!("L")
ylabel!("free energy from <L|R>")
savefig(filenameR*"LR_free_energy.pdf")