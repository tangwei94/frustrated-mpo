using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie
using QuadGK 

include("../utils.jl");

βc = asinh(1) / 2
k = 1 / (sinh(2*βc))^2
f_exact = log(2) / 2 + (1/2/pi) * quadgk(θ-> log(cosh(2*βc)*cosh(2*βc) + (1/k)*sqrt(1+k^2-2*k*cos(2*θ))), 0, pi)[1]

T = tensor_square_ising(βc)

σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
function genP(τ::Real, O::AbstractTensorMap)
    P = add_util_leg(exp(-τ*O))
    ℙ = mpo_gen(1, P, :inf)
    return P, ℙ
end

# VUMPS for original Hermitian problem
𝕋0 = mpo_gen(1, T, :inf)

function f_normality(τ::Real)
    ℙ = genP(τ, σx)[2]
    ℙinv = genP(-τ, σx)[2]

    𝕋1 = ℙ * 𝕋0 * ℙinv
    𝕋1dag = ℙinv * 𝕋0 * ℙ 

    ϕ1 = convert(InfiniteMPS, 𝕋1*𝕋1dag)
    ϕ2 = convert(InfiniteMPS, 𝕋1dag*𝕋1)

    return norm(dot(ϕ1, ϕ2)), 𝕋1, 𝕋1dag
end

function VUMPS_history(τ::Real, χ::Int, maxiter::Int)
    normality, 𝕋1, 𝕋1dag = f_normality(τ)
    ψt0 = InfiniteMPS([ℂ^2], [ℂ^χ])

    f_history = Float64[]
    galerkin_history = Float64[]
    variance_history = Float64[]
    ℙinv2 = genP(-τ*2, σx)[2]
    function finalize1(iter,state,H,envs)
        st = convert(InfiniteMPS,state)
        sb = ℙinv2 * st 
        f1 = log(norm(dot(sb, 𝕋1, st))) - log(norm(dot(sb, st)))
        push!(f_history, f1)
        push!(galerkin_history, MPSKit.calc_galerkin(state, envs))
        push!(variance_history, log(norm(dot(st, 𝕋1dag*𝕋1, st) / dot(st, 𝕋1dag, st) / dot(st, 𝕋1, st))))
        @show length(f_history), length(galerkin_history)
        return (state, envs)
    end

    optim_alg = VUMPS(tol_galerkin=1e-12, maxiter=maxiter, finalize=finalize1)
    ψt, envt, _ = leading_boundary(ψt0, 𝕋1, optim_alg)

    return normality, f_history, galerkin_history, variance_history
end

normality1, f_history1, galerkin_history1, variance_history1 = VUMPS_history(0.1, 32, 200)
normality2, f_history2, galerkin_history2, variance_history2 = VUMPS_history(0.2, 32, 200)
normality3, f_history3, galerkin_history3, variance_history3 = VUMPS_history(0.3, 32, 200)
normality4, f_history4, galerkin_history4, variance_history4 = VUMPS_history(0.4, 32, 200)

@save "square_ising/data/badly_gauged-VUMPS-histories_1.jld2" normality1 f_history1 galerkin_history1 variance_history1 
@save "square_ising/data/badly_gauged-VUMPS-histories_2.jld2" normality2 f_history2 galerkin_history2 variance_history2 
@save "square_ising/data/badly_gauged-VUMPS-histories_3.jld2" normality3 f_history3 galerkin_history3 variance_history3 
@save "square_ising/data/badly_gauged-VUMPS-histories_4.jld2" normality4 f_history4 galerkin_history4 variance_history4 
@load "square_ising/data/badly_gauged-VUMPS-histories_1.jld2" normality1 f_history1 galerkin_history1 variance_history1 
@load "square_ising/data/badly_gauged-VUMPS-histories_2.jld2" normality2 f_history2 galerkin_history2 variance_history2 
@load "square_ising/data/badly_gauged-VUMPS-histories_3.jld2" normality3 f_history3 galerkin_history3 variance_history3 
@load "square_ising/data/badly_gauged-VUMPS-histories_4.jld2" normality4 f_history4 galerkin_history4 variance_history4 

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 750))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:length(f_history1), abs.(f_history1 .- f_exact) ./ f_exact, label=L"\tau=0.1")
lines!(ax1, 1:length(f_history2), abs.(f_history2 .- f_exact) ./ f_exact, label=L"\tau=0.2")
lines!(ax1, 1:length(f_history3), abs.(f_history3 .- f_exact) ./ f_exact, label=L"\tau=0.3")
lines!(ax1, 1:length(f_history4), abs.(f_history4 .- f_exact) ./ f_exact, label=L"\tau=0.4")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{convergence measure}", yscale=log10)
lines!(ax2, 1:length(f_history1), galerkin_history1, label=L"\tau=0.1")
lines!(ax2, 1:length(f_history2), galerkin_history2, label=L"\tau=0.2")
lines!(ax2, 1:length(f_history3), galerkin_history3, label=L"\tau=0.3")
lines!(ax2, 1:length(f_history4), galerkin_history4, label=L"\tau=0.4")
@show fig 

ax3 = Axis(fig[3, 1], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10)
lines!(ax3, 1:length(f_history1), variance_history1, label=L"\tau=0.1")
lines!(ax3, 1:length(f_history2), variance_history2, label=L"\tau=0.2")
lines!(ax3, 1:length(f_history3), variance_history3, label=L"\tau=0.3")
lines!(ax3, 1:length(f_history4), variance_history4, label=L"\tau=0.4")
#axislegend(ax3)
@show fig 
