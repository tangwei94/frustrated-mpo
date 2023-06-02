using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

@load "gauge_AF_triangular_ising/data/VUMPS_data.jld2" ψs3 ψs2 ψs1 

A = ψs1[2].AL[1]
L = 18
ψfin = circular_mps(A, L);

left_virtualspace(ψfin, L÷2)

function meas_domainwalls()
    σzs = perfect_sampling(ψfin)
    N = num_domain_wall(σzs, :frstr, :pbc)
    return N
end

meas_domainwalls()

Ns = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_domainwalls()
end

bins = findmax(Ns)[1] - findmin(Ns)[1]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"X", ylabel=L"N")
xlims!(ax1, (0, L))
hist!(ax1, Ns; bins=bins)
@show fig