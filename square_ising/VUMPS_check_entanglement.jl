using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");

βc = asinh(1) / 2

T = tensor_square_ising(βc)

σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
function generate_P(τ::Real, O::AbstractTensorMap)
    P = add_util_leg(exp(-τ*O))
    ℙ = mpo_gen(1, P, :inf)
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

EEs = [real(entropy(ψ)[1]) for ψ in ψs]
χs = 8:8:64

function RDM_after_gauge(ψ, τ, O)
    expO = exp(-τ * O)
    ψt = ψ.AL[1]
    @tensor Pψt[-1 -2; -3] := ψt[-1 1 ; -3] * expO[-2 1]

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

    return rdm, real(EE)
end

for ix in 1:8
    @show RDM_after_gauge(ψs[ix], 0, σz)[2] - EEs[ix] |> norm
end

EE1s = [RDM_after_gauge(ψ, 0.2, σz)[2] for ψ in ψs]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\ln \chi", ylabel=L"S")
lines1 = lines!(ax1, log.(χs), abs.(EEs) .+ 1e-16, marker=:circle, markersize=5, label=L"\tau=0")
lines1 = lines!(ax1, log.(χs), abs.(EE1s) .+ 1e-16, marker=:circle, markersize=5, label=L"\tau=0.5")
axislegend(ax1; position=:rb)
@show fig
save("square_ising/data/EEs-local-gauge.pdf", fig)

τs = -1:0.05:1
EExs = []
EEzs = []
for ix in 1:8
    push!(EExs, [RDM_after_gauge(ψs[ix], τ, σx)[2] for τ in τs])
    push!(EEzs, [RDM_after_gauge(ψs[ix], τ, σz)[2] for τ in τs])
    @show left_virtualspace(ψs[ix], 1)
end
@save "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs EExs EEzs τs
@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs EExs EEzs τs

EExs

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"S_E")
for ix in 2:8
    lines!(ax1, τs, abs.(EExs[ix]) .+ 1e-16, label="σˣ, $(left_virtualspace(ψs[ix], 1))")
end
ax2 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"S_E")
for ix in 2:8
    lines!(ax2, τs, abs.(EEzs[ix]) .+ 1e-16, label="σᶻ, $(left_virtualspace(ψs[ix], 1))")
end
#liney = lines!(ax1, τs, abs.(EEys) .+ 1e-16, label=L"\sigma^y")
#linez = lines!(ax1, τs, abs.(EEzs) .+ 1e-16, label=L"\sigma^z")
axislegend(ax1)
axislegend(ax2)
@show fig

save("square_ising/data/VUMPS-EE_vs_tau.pdf", fig)