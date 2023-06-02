using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

f_exact = 0.3230659669

ψ1 = InfiniteMPS([ℂ^2], [ℂ^16])
ψ2 = InfiniteMPS([ℂ^2], [ℂ^16])
optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=1000) 
results_16 = map(1:100) do ix
    global ψ1, ψ2
    ψ1, _ = approximate(ψ1, (𝕋, ψ1), optim_alg1)
    ψ2, _ = approximate(ψ2, (𝕋dag, ψ2), optim_alg1)
    f = log(dot(ψ2, 𝕋, ψ1) / dot(ψ2, ψ1))
    return ψ1, ψ2, f
end

ψ1 = InfiniteMPS([ℂ^2], [ℂ^32])
ψ2 = InfiniteMPS([ℂ^2], [ℂ^32])
results_32 = map(1:100) do ix
    global ψ1, ψ2
    ψ1, _ = approximate(ψ1, (𝕋, ψ1), optim_alg1)
    ψ2, _ = approximate(ψ2, (𝕋dag, ψ2), optim_alg1)
    f = log(dot(ψ2, 𝕋, ψ1) / dot(ψ2, ψ1))
    return ψ1, ψ2, f
end

ψ1 = InfiniteMPS([ℂ^2], [ℂ^64])
ψ2 = InfiniteMPS([ℂ^2], [ℂ^64])
results_64 = map(1:100) do ix
    global ψ1, ψ2
    ψ1, _ = approximate(ψ1, (𝕋, ψ1), optim_alg1)
    ψ2, _ = approximate(ψ2, (𝕋dag, ψ2), optim_alg1)
    f = log(dot(ψ2, 𝕋, ψ1) / dot(ψ2, ψ1))
    return ψ1, ψ2, f
end

extract_result(results) = (
    [result[1] for result in results],
    [result[2] for result in results],
    [result[3] for result in results]
)

ψ1s_16, ψ2s_16, fs_16 = extract_result(results_16)
ψ1s_32, ψ2s_32, fs_32 = extract_result(results_32)
ψ1s_64, ψ2s_64, fs_64 = extract_result(results_64)

@save "gauge_AF_triangular_ising/data/vomps_chi16_results.jld2" ψ1s_16 ψ2s_16 fs_16
@save "gauge_AF_triangular_ising/data/vomps_chi32_results.jld2" ψ1s_32 ψ2s_32 fs_32
@save "gauge_AF_triangular_ising/data/vomps_chi64_results.jld2" ψ1s_64 ψ2s_64 fs_64

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
CairoMakie.scatter!(ax1, 1:100, abs.(fs_16 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"χ=16")
CairoMakie.scatter!(ax1, 1:100, abs.(fs_32 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"χ=32")
CairoMakie.scatter!(ax1, 1:100, abs.(fs_64 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"χ=64")
axislegend(ax1)
@show fig 
save("gauge_AF_triangular_ising/data/VOMPS_plot.pdf", fig)

