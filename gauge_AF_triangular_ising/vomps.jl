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

extract_result(results) = (
    [result[1] for result in results],
    [result[2] for result in results],
    [result[3] for result in results],
    [result[4] for result in results],
    [result[5] for result in results]
)

ψ1 = InfiniteMPS([ℂ^2], [ℂ^32])
ψ2 = InfiniteMPS([ℂ^2], [ℂ^32])
optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=10000) 
results_32 = map(1:100) do ix
    global ψ1, ψ2
    ψ1, _ = approximate(ψ1, (𝕋, ψ1), optim_alg1)
    ψ2, _ = approximate(ψ2, (𝕋dag, ψ2), optim_alg1)
    var1 = log(dot(ψ1, 𝕋dag*𝕋, ψ1) / dot(ψ1, 𝕋, ψ1) / dot(ψ1, 𝕋dag, ψ1))
    var2 = log(dot(ψ2, 𝕋*𝕋dag, ψ2) / dot(ψ2, 𝕋, ψ2) / dot(ψ2, 𝕋dag, ψ2))
    f = log(dot(ψ2, 𝕋, ψ1) / dot(ψ2, ψ1))
    return ψ1, ψ2, f, var1, var2
end;
ψ1s_32, ψ2s_32, fs_32, vars1_32, vars2_32 = extract_result(results_32);
_, ix1 = findmin(real.(vars1_32))
_, ix2 = findmin(real.(vars2_32))

@save "gauge_AF_triangular_ising/data/vomps_chi32_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:100, abs.(fs_32 .- f_exact) ./ f_exact, label=L"χ=32")
text!(ax1, 0, 10^1, text=L"\text{(a)}", align=(:left, :top))
@show fig 

ax2 = Axis(fig[1, 2], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10)
lines!(ax2, 1:100, norm.(vars1_32), label=L"|\psi^{\mathrm{R}}\rangle")
lines!(ax2, 1:100, norm.(vars2_32), label=L"|\psi^{\mathrm{L}}\rangle")
text!(ax2, 0, 10^(0), text=L"\text{(b)}", align=(:left, :top))
axislegend(ax2, position=:rt)
save("gauge_AF_triangular_ising/data/fig-VOMPS_plot.pdf", fig)
@show fig 