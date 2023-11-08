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

indices = ["000", "025", "050", "075", "100", "125", "150", "175"];
τs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
ψRs_x = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    return VOMPS_results[1][end]
end; 
ψRs_z = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_z_$(index).jld2" VOMPS_results
    return VOMPS_results[1][end]
end; 

τ2s = 0:0.05:2
for (index, τ, ψR) in zip(indices, τs, ψRs_x)
    vars = map(τs) do τ2
        @show τ, τ2
        variance_at_τ(ψR, τ, τ2)
    end;
    @save "square_ising/data/VOMPS_variances_x_$(index).jld2" ψR τ τ2s vars
end
for (index, τ, ψR) in zip(indices, τs, ψRs_z)
    vars = map(τs) do τ2
        @show τ, τ2
        variance_at_τ(ψR, τ, τ2, σz)
    end;
    @save "square_ising/data/VOMPS_variances_z_$(index).jld2" ψR τ τ2s vars
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"τ", ylabel=L"\text{variance}", yscale=log10)
for (index, τ) in zip(indices[1:2:end], τs[1:2:end])
    @load "square_ising/data/VOMPS_variances_x_$(index).jld2" ψR τ τ2s vars
    scatterlines!(ax1, τs, abs.(vars), linestyle=:dash, label="$(τ)")
end
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel=L"τ", ylabel=L"\text{variance}", yscale=log10)
for (index, τ) in zip(indices[1:2:end], τs[1:2:end])
    @load "square_ising/data/VOMPS_variances_z_$(index).jld2" ψR τ τ2s vars
    scatterlines!(ax2, τs, abs.(vars), linestyle=:dash, label="$(τ)")
end
axislegend(ax2)
@show fig 

indices = ["000", "025", "050", "075", "100", "125", "150", "175"];
τs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
ψRs_x = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    return VOMPS_results[1][end]
end; 
ψRs_z = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_z_$(index).jld2" VOMPS_results
    return VOMPS_results[1][end]
end; 

τ2s = 0:0.05:2
for (index, τ, ψR) in zip(indices, τs, ψRs_x)
    vars = map(τs) do τ2
        @show τ, τ2
        variance_at_τ(ψR, τ, τ2)
    end;
    @save "square_ising/data/VOMPS_variances_x_$(index).jld2" ψR τ τ2s vars
end
for (index, τ, ψR) in zip(indices, τs, ψRs_z)
    vars = map(τs) do τ2
        @show τ, τ2
        variance_at_τ(ψR, τ, τ2, σz)
    end;
    @save "square_ising/data/VOMPS_variances_z_$(index).jld2" ψR τ τ2s vars
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"τ", ylabel=L"\text{variance}", yscale=log10)
for (index, τ) in zip(indices[1:2:end], τs[1:2:end])
    @load "square_ising/data/VOMPS_variances_x_$(index).jld2" ψR τ τ2s vars
    scatterlines!(ax1, τs, abs.(vars), linestyle=:dash, label="$(τ)")
end
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel=L"τ", ylabel=L"\text{variance}", yscale=log10)
for (index, τ) in zip(indices[1:2:end], τs[1:2:end])
    @load "square_ising/data/VOMPS_variances_z_$(index).jld2" ψR τ τ2s vars
    scatterlines!(ax2, τs, abs.(vars), linestyle=:dash, label="$(τ)")
end
axislegend(ax2)
@show fig 