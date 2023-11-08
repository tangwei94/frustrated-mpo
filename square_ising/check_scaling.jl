# check whether the variance is gauge invariant

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
    ℙ = genP(τ, O)[2]
    ℙinv = genP(-τ, O)[2]

    𝕋1 = ℙ * 𝕋0 * ℙinv
    𝕋1dag = ℙinv * 𝕋0 * ℙ 

    a1 = 𝕋1.opp[1]
    a2 = 𝕋1dag.opp[1]

    normality = real(mpo_ovlp(a1, a2)[1][1] * mpo_ovlp(a2, a1)[1][1] / mpo_ovlp(a1, a1)[1][1] / mpo_ovlp(a2, a2)[1][1])

    return normality, 𝕋1, 𝕋1dag
end

function variance_at_τ(ψ1, τ1, τ2, O=σx)
    _, 𝕋2, 𝕋2dag = f_normality(τ2, O)

    _, ℙ = generate_P((τ2-τ1), O)

    var = log(norm(dot(ψ1, ℙ*𝕋2dag*𝕋2*ℙ, ψ1) * dot(ψ1, ℙ*ℙ, ψ1) / dot(ψ1, ℙ*𝕋2dag*ℙ, ψ1) / dot(ψ1, ℙ*𝕋2*ℙ, ψ1)))
    return var
end

@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs

entropy(ψs[end], 1)
calc_entropy(ψs[end], ψs[end])

τ = 0.5
Px, Pxinv = exp(-0.5*σx), exp(0.5*σx)
ℙx, ℙxinv = DenseMPO(add_util_leg(Px)), DenseMPO(add_util_leg(Pxinv))

calc_entropy(ψs[end], ℙxinv, ℙx, ψs[end])[2]

calc_correlation_length(ψs[end], ψs[end])
calc_correlation_length(ψs[end], ℙxinv*ℙx, ψs[end])

indices = ["000", "025", "050", "075", "100", "125", "150", "175"];
τs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75];

for (index, τ) in zip(indices, τs)
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    EEs = Float64[]
    corrs = Float64[]
    for (ψR, ψL) in zip(VOMPS_results[1][250:250:end], VOMPS_results[2][250:250:end])
        @show index, MPSKit._firstspace(ψR.AL[1])
        push!(EEs, calc_entropy(ψL, ψR))
        push!(corrs, calc_correlation_length(ψL, ψR))
    end
    @save "square_ising/data/badly_gauged-VOMPS-correlations_$(index).jld2" EEs corrs
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"ln \xi", ylabel=L"S_E")
for (index, τ) in zip(indices[1:2:end-1], τs[1:2:end-1])
    @load "square_ising/data/badly_gauged-VOMPS-correlations_$(index).jld2" EEs corrs
    scatterlines!(ax1, log.(corrs), EEs, linestyle=:dot, label="$(τ)")
end
lines!(ax1, 1:6, (1/12)*(1:6), linestyle=:dash, label=L"(1/12) \ln ξ")
axislegend(ax1, position=:rb)
@show fig
