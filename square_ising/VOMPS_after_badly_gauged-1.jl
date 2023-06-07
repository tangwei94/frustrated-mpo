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

function VOMPS_history(τ::Real, χ::Int)
    normality1, 𝕋1, 𝕋1dag = f_normality(τ)
    @show normality1
    optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=400) 
    ℙinv2 = genP(-τ*2, σx)[2]
    ψ1 = InfiniteMPS([ℂ^2], [ℂ^χ])

    ψ1s = typeof(ψ1)[]
    fs = Float64[]
    vars = Float64[]
    for ix in 1:1000
        ψ2 = ℙinv2 * ψ1 
        ψ1, _ = approximate(ψ1, (𝕋1, ψ1), optim_alg1)
        f = real(log(dot(ψ2, 𝕋1, ψ1) / dot(ψ2, ψ1)))
        var = log(norm(dot(ψ1, 𝕋1dag*𝕋1, ψ1) / dot(ψ1, 𝕋1dag, ψ1) / dot(ψ1, 𝕋1, ψ1)))
        push!(ψ1s, ψ1)
        push!(fs, f)
        push!(vars, var)
    end
    return ψ1s, fs, vars
end
ψ0s, f0s, var0s = VOMPS_history(0, 32)
ψ1s, f1s, var1s = VOMPS_history(0.1, 32)
ψ2s, f2s, var2s = VOMPS_history(0.2, 32)
ψ3s, f3s, var3s = VOMPS_history(0.3, 32)
ψ4s, f4s, var4s = VOMPS_history(0.4, 32)

@save "square_ising/data/badly_gauged-VOMPS-histories_0.jld2" ψ0s f0s var0s 
@save "square_ising/data/badly_gauged-VOMPS-histories_1.jld2" ψ1s f1s var1s 
@save "square_ising/data/badly_gauged-VOMPS-histories_2.jld2" ψ2s f2s var2s 
@save "square_ising/data/badly_gauged-VOMPS-histories_3.jld2" ψ3s f3s var3s 
@save "square_ising/data/badly_gauged-VOMPS-histories_4.jld2" ψ4s f4s var4s 
@load "square_ising/data/badly_gauged-VOMPS-histories_0.jld2" ψ0s f0s var0s 
@load "square_ising/data/badly_gauged-VOMPS-histories_1.jld2" ψ1s f1s var1s 
@load "square_ising/data/badly_gauged-VOMPS-histories_2.jld2" ψ2s f2s var2s 
@load "square_ising/data/badly_gauged-VOMPS-histories_3.jld2" ψ3s f3s var3s 
@load "square_ising/data/badly_gauged-VOMPS-histories_4.jld2" ψ4s f4s var4s 

@load "square_ising/data/VUMPS_hermitian_betac.jld2" ψs fs
MPSKit.left_virtualspace(ψs[4], 1)
ferr0 = abs(-fs[4]*βc - f_exact)/f_exact

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:1000, abs.(f0s .- f_exact) ./ f_exact, label=L"τ=0.0")
lines!(ax1, 1:1000, abs.(f1s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:1000, abs.(f2s .- f_exact) ./ f_exact, label=L"τ=0.2")
lines!(ax1, 1:1000, abs.(f3s .- f_exact) ./ f_exact, label=L"τ=0.3")
lines!(ax1, 1:1000, abs.(f4s .- f_exact) ./ f_exact, label=L"τ=0.4")
lines!(ax1, 1:1000, fill(ferr0, 1000), linestyle=:dash, label=L"\text{hermitian VUMPS}")
#axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10)
lines!(ax2, 1:1000, var1s, label=L"τ=0.0")
lines!(ax2, 1:1000, var1s, label=L"τ=0.1")
lines!(ax2, 1:1000, var2s, label=L"τ=0.2")
lines!(ax2, 1:1000, var3s, label=L"τ=0.3")
lines!(ax2, 1:1000, var4s, label=L"τ=0.4")
axislegend(ax2)
@show fig

# not easy to see the difference. change strategy