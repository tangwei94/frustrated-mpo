# calculate the entanglement entropy of the eigenstate after the gauge transformation
# using the VUMPS result ψ0 to do the computation.
# output figures in the manuscript.

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
𝕋 = mpo_gen(1, T, :inf)

expand_alg = OptimalExpand(truncdim(8))
optim_alg = VUMPS(tol_galerkin=1e-12, maxiter=10000)
ψs = InfiniteMPS[]
fs = Float64[]
ψ0 = InfiniteMPS([ℂ^2], [ℂ^8])
for ix in 1:8
    ψ, env, _ = leading_boundary(ψ0, 𝕋, optim_alg)
    ψ0, _ = changebonds(ψ, 𝕋, expand_alg, env)
    f1 = -log(norm(dot(ψ, 𝕋, ψ)))/βc

    @show space(ψ.AL[1]), f1 
    push!(fs, f1)
    push!(ψs, ψ)
end

@save "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs
@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs

# original EEs
EEs = [real(entropy(ψ)[1]) for ψ in ψs]
χs = 8:8:64

function RDM_after_gauge(ψ, τ, O)
    expO = exp(-τ * O)
    expO_rv = exp(τ * O)
    ψt = ψ.AL[1]
    @tensor Pψt[-1 -2; -3] := ψt[-1 1 ; -3] * expO[-2 1]
    @tensor Prvψt[-1 -2; -3] := ψt[-1 1 ; -3] * expO_rv[-2 1]

    transferR = TransferMatrix(Pψt)
    transferL = TransferMatrix(Pψt, nothing, Pψt, true)
    sp = left_virtualspace(ψ, 1)
    v0 = TensorMap(rand, ComplexF64, sp, sp)

    val, ρr, _ = eigsolve(transferR, v0, 1, :LM)
    val, ρl, _ = eigsolve(transferL, v0, 1, :LM)
    ρr = ρr[1] / ρr[1][1] * abs(ρr[1][1])
    ρl = ρl[1] / ρl[1][1] * abs(ρl[1][1])

    Λr, Vr = eigh(ρr)
    Λl, Vl = eigh(ρl)

    C = sqrt(Λl) * Vl' * Vr * sqrt(Λr)
    rdm = C * C' / tr(C * C')
    EE = -tr(MPSKit.safe_xlogx(rdm))

    printstyled("$(sp), τ=$(τ)\n", color=:red)
    return rdm, real(EE)
end

# entanglement entropy
τs = -2.0:0.1:2.0
τs = -1.0:0.1:1.0
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ga = fig[1:5, 1:6] = GridLayout()
gb = fig[6, 2:4] = GridLayout()
ax1 = Axis(ga[1, 1], xlabel=L"\tau", ylabel=L"\text{S^{\mathrm{R}}}", yscale=log10)
for (χ, ψ) in zip(χs, ψs)
    if χ % 16 != 0 continue end
    EEs = map(τs) do τ
        _, EE = RDM_after_gauge(ψ, τ, σx)
        return EE
    end
    lines!(ax1, τs, abs.(EEs), label=latexstring("\$\\chi=$(χ)\$") )
end
ax2 = Axis(ga[1, 2], xlabel=L"\tau", yscale=log10 )
for (χ, ψ) in zip(χs, ψs)
    if χ % 16 != 0 continue end
    EEs = map(τs) do τ
        _, EE = RDM_after_gauge(ψ, τ, σz)
        return EE
    end
    lines!(ax2, τs, abs.(EEs), label=latexstring("\$\\chi=$(χ)\$") )
end
Legend(gb[1, 1], ax1, nbanks=5)
save("square_ising/data/fig-EEs-local-gauge.pdf", fig)
@show fig

# compare entanglement spectrum 
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ga = fig[1:5, 1] = GridLayout()
gb = fig[6, 1] = GridLayout()
ax1 = Axis(ga[1, 1], xlabel=L"\text{index}", ylabel=L"\text{spectrum of ρ_A^{\mathrm{RR}}}", yscale=log10)
for τ in 0:0.5:2.0 
    RDM_R, = RDM_after_gauge(ψs[end], τ, σx);
    ESRx = eigen(RDM_R)[1].data |> diag;
    ESRx .*= (sign.(ESRx) .+ 1) ./ 2
    scatter!(ax1, 64:-1:1, abs.(ESRx), marker=:circle, markersize=10, label=latexstring("\$\\tau=$(τ)\$") )
end
text!(ax1, 0., 1, text=L"\text{(a)}", align=(:left, :bottom))
ylims!(ax1, (1e-16, 10^1.5))
#axislegend(ax1; position=:rt)
@show fig

ax2 = Axis(ga[1, 2], xlabel=L"\text{index}", yticklabelsvisible=false, yscale=log10)
for τ in 0:0.5:2.0 
    RDM_R, = RDM_after_gauge(ψs[end], τ, σz);
    ESRx = eigen(RDM_R)[1].data |> diag;
    ESRx .*= (sign.(ESRx) .+ 1) ./ 2
    scatter!(ax2, 64:-1:1, abs.(ESRx), marker=:circle, markersize=10, label=latexstring("\$\\tau=$(τ)\$") )
end
text!(ax2, 0., 1, text=L"\text{(b)}", align=(:left, :bottom))
ylims!(ax2, (1e-16, 10^1.5))
#axislegend(ax2; position=:rt)
@show fig

Legend(gb[1, 1], ax1, nbanks=5)

save("square_ising/data/fig-ESs-local-gauge.pdf", fig)
@show fig

# M matrix
ψ0 = ψs[end-1]
τ, M = 0.5, σx
ψl = DenseMPO([add_util_leg(exp(-τ*M))]) * ψ0;
ψr = DenseMPO([add_util_leg(exp(τ*M))]) * ψ0;

scattering = importance_scattering_L(ψl, ψr); 
fig, ax, hm = heatmap(norm.(scattering.data); colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues, figure = (; resolution=(600, 500)) , axis = (; xlabel=L"i", ylabel=L"j"))

for (τ, M, locx, locy) in zip([1.0, 0.5, 1.0], [σx, σz, σz], [1, 2, 2], [2, 1, 2])
    ψl = DenseMPO([add_util_leg(exp(-τ*M))]) * ψ0;
    ψr = DenseMPO([add_util_leg(exp(τ*M))]) * ψ0;

    scattering = importance_scattering_L(ψl, ψr);
    ax = Axis(fig[locx, locy], xlabel=L"i", ylabel=L"j") 
    heatmap!(ax, norm.(scattering.data), colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues)
end
Colorbar(fig[:, end+1], hm)
save("square_ising/data/fig-M-matrix.pdf", fig)

@show fig
