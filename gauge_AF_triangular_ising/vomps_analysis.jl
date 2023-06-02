using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

@load "gauge_AF_triangular_ising/data/vomps_chi16_results.jld2" ψ1s_16 ψ2s_16 fs_16
@load "gauge_AF_triangular_ising/data/vomps_chi32_results.jld2" ψ1s_32 ψ2s_32 fs_32
@load "gauge_AF_triangular_ising/data/vomps_chi64_results.jld2" ψ1s_64 ψ2s_64 fs_64

EEs_16 = [real(entropy(ψ)[1]) for ψ in ψ1s_16]
EEs_32 = [real(entropy(ψ)[1]) for ψ in ψ1s_32]
EEs_64 = [real(entropy(ψ)[1]) for ψ in ψ1s_64]

@load "gauge_AF_triangular_ising/data/VUMPS_data.jld2" ψsinf ψs3 ψs2 ψs1 fsinf fs3 fs2 fs1
@show [MPSKit.virtualspace(ψ, 1) for ψ in ψsinf] 

EEs_0 = [real(entropy(ψ)[1]) for ψ in ψsinf]
EEs_0[[2,4,8]]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"S")
lines!(ax1, 1:100, abs.(EEs_16) .+ 1e-16, label=L"\chi=16")
lines!(ax1, 1:100, fill(EEs_0[2], 100), linestyle=:dash, label=L"\chi=16 \text{ hermitian}")
lines!(ax1, 1:100, abs.(EEs_32) .+ 1e-16, label=L"\chi=32")
lines!(ax1, 1:100, fill(EEs_0[4], 100), linestyle=:dash, label=L"\chi=32 \text{ hermitian}")
lines!(ax1, 1:100, abs.(EEs_64) .+ 1e-16, label=L"\chi=64")
lines!(ax1, 1:100, fill(EEs_0[8], 100), linestyle=:dash, label=L"\chi=64 \text{ hermitian}")
axislegend(ax1; position=:rt)
@show fig
save("gauge_AF_triangular_ising/data/triangleAF-VOMPS-entanglement_entropy.pdf", fig)

# domain wall configuration 
L = 18
A_chi32_step40 = ψ1s_32[40].AL[1];
ϕ1_chi32_step40 = circular_mps(A_chi32_step40, L);

B_chi32_step40 = ψ2s_32[40].AL[1];
ϕ2_chi32_step40 = circular_mps(B_chi32_step40, L);

function meas_domainwalls(ϕ)
    σzs = perfect_sampling(ϕ)
    N = num_domain_wall(σzs, :frstr, :pbc)
    return N
end

Ns_1 = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_domainwalls(ϕ1_chi32_step40)
end

Ns_2 = map(1:50*L) do ix
    ix % 10 == 0 && (@show ix) 
    return meas_domainwalls(ϕ2_chi32_step40)
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"\text{number of domain walls}", ylabel=L"N")
xlims!(ax1, (1, L))
hist!(ax1, Ns_1; bins=1:L)
hist!(ax1, Ns_2; bins=1:L)
@show fig

save("gauge_AF_triangular_ising/data/triangleAF-VOMPS-domain_wall_distribution.pdf", fig)

# variance
Tn = tensor_triangular_AF_ising_adapted()
Tn_dag = mpotensor_dag(Tn)
𝕋n = mpo_gen(1, Tn, :inf)
𝕋n_dag = mpo_gen(1, Tn_dag, :inf)

ψ1 = ψsinf[2]
ψ2 = ψsinf[4]
ψ3 = ψsinf[8]

var_n_16 = -real(log(dot(ψ1, 𝕋n, ψ1) * dot(ψ1, 𝕋n_dag, ψ1) / dot(ψ1, 𝕋n_dag * 𝕋n, ψ1)))
var_n_32 = -real(log(dot(ψ2, 𝕋n, ψ2) * dot(ψ2, 𝕋n_dag, ψ2) / dot(ψ2, 𝕋n_dag * 𝕋n, ψ2)))
var_n_64 = -real(log(dot(ψ3, 𝕋n, ψ3) * dot(ψ3, 𝕋n_dag, ψ3) / dot(ψ3, 𝕋n_dag * 𝕋n, ψ3)))

variance_T(ψ::InfiniteMPS) = -real(log(dot(ψ, 𝕋, ψ) * dot(ψ, 𝕋dag, ψ) / dot(ψ, 𝕋dag * 𝕋, ψ)))

var_16 = variance_T.(ψ1s_16)
var_32 = variance_T.(ψ1s_32)
var_64 = variance_T.(ψ1s_64)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"N", yscale=log10)
lines!(ax1, 1:100, var_16, label=L"\chi=16")
lines!(ax1, 1:100, fill(var_n_16, 100), linestyle=:dash, label=L"\chi=16 \text{ normal}")
lines!(ax1, 1:100, var_32, label=L"\chi=32")
lines!(ax1, 1:100, fill(var_n_32, 100), linestyle=:dash, label=L"\chi=32 \text{ normal}")
lines!(ax1, 1:100, var_64, label=L"\chi=64")
lines!(ax1, 1:100, fill(var_n_64, 100), linestyle=:dash, label=L"\chi=64 \text{ normal}")
axislegend(ax1; position=:rt)
@show fig

save("gauge_AF_triangular_ising/data/triangleAF-VOMPS-var.pdf", fig)
