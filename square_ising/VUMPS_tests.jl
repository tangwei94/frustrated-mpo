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

τ = -0.1
ℙ = generate_P(τ)[2]
ψτs = Ref(ℙ) .* ψs;

EE1s = [real(entropy(ψτ)[1]) for ψτ in ψτs]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\ln \chi", ylabel=L"S")
lines1 = lines!(ax1, log.(χs), abs.(EEs) .+ 1e-16, marker=:circle, markersize=5, label=L"\tau=0")
lines1 = lines!(ax1, log.(χs), abs.(EE1s) .+ 1e-16, marker=:circle, markersize=5, label=L"\tau=0.5")
axislegend(ax1; position=:rb)
@show fig
save("square_ising/data/EEs-local-gauge.pdf", fig)

ψ0 = ψs[4]
EExs = Float64[]
EEzs = Float64[]
EEys = Float64[]
xs, ys, zs = Float64[], Float64[], Float64[]
τs = -1:0.1:1

for τ in τs
    ℙx = generate_P(τ, σx)[2]
    ϕx = ℙx * ψ0
    push!(EExs, real(entropy(ϕx)[1]))
    ℙz = generate_P(τ, σz)[2]
    ϕz = ℙz * ψ0
    push!(EEzs, real(entropy(ϕz)[1]))
    ℙy = generate_P(τ, σy)[2]
    ϕy = ℙy * ψ0
    push!(EEys, real(entropy(ϕy)[1]))
    push!(xs, real(expectation_value(ϕx, σx)[1]))
    push!(ys, real(expectation_value(ϕy, σy)[1]))
    push!(zs, real(expectation_value(ϕz, σz)[1]))
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"S")
scatterx = scatter!(ax1, τs, abs.(EExs) .+ 1e-16, label=L"\sigma^x")
scattery = scatter!(ax1, τs, abs.(EEys) .+ 1e-16, label=L"\sigma^y")
scatterz = scatter!(ax1, τs, abs.(EEzs) .+ 1e-16, label=L"\sigma^z")
axislegend(ax1; position=:rb)
@show fig

ax2 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"\langle O \rangle")
scatterx = scatter!(ax2, τs, xs, label=L"O = \sigma^x")
scattery = scatter!(ax2, τs, ys, label=L"O = \sigma^y")
scatterz = scatter!(ax2, τs, zs, label=L"O = \sigma^z")
axislegend(ax2; position=:rt)
@show fig