using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie, LaTeXStrings

include("../utils.jl");

βc = asinh(1) / 2

T = tensor_square_ising(βc)

σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
function generate_P(τ::Real, O::AbstractTensorMap)
    P = add_util_leg(exp(-τ*O))
    ℙ = DenseMPO([P])
    return P, ℙ
end

# VUMPS for original Hermitian problem
𝕋0 = mpo_gen(1, T, :inf)

function f_normality(τ::Real, O::AbstractTensorMap)
    ℙ = generate_P(τ, O)[2]
    ℙinv = generate_P(-τ, O)[2]

    𝕋1 = ℙ * 𝕋0 * ℙinv
    𝕋1dag = ℙinv * 𝕋0 * ℙ 

    a1 = 𝕋1.opp[1]
    a2 = 𝕋1dag.opp[1]

    normality = real(mpo_ovlp(a1, a2)[1][1] * mpo_ovlp(a2, a1)[1][1] / mpo_ovlp(a1, a1)[1][1] / mpo_ovlp(a2, a2)[1][1])

    return normality, 𝕋1, 𝕋1dag
end

τ2s = 0:0.05:2
normalities_z = map(τ2s) do τ2
    return f_normality(τ2, σz)[1]
end
normalities_x = map(τ2s) do τ2
    return f_normality(τ2, σx)[1]
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 300))
ax1 = Axis(fig[1, 1], xlabel=L"τ", ylabel=L"\text{normality measure}")
lines!(ax1, τ2s, normalities_x, label=L"Q=σ^x")
lines!(ax1, τ2s, normalities_z, label=L"Q=σ^z")
axislegend(ax1, position=:rt)
save("square_ising/data/fig-normalities-change.pdf", fig)
@show fig 