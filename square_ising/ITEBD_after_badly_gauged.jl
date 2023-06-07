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

function f_normality(τ::Real, O::AbstractTensorMap)
    ℙ = genP(τ, O)[2]
    ℙinv = genP(-τ, O)[2]

    𝕋1 = ℙ * 𝕋0 * ℙinv
    𝕋1dag = ℙinv * 𝕋0 * ℙ 

    ϕ1 = convert(InfiniteMPS, 𝕋1*𝕋1dag)
    ϕ2 = convert(InfiniteMPS, 𝕋1dag*𝕋1)

    return norm(dot(ϕ1, ϕ2)), 𝕋1, 𝕋1dag
end

function iTEBD_history(τ::Real, O::AbstractTensorMap,  err::Real)
    normality1, 𝕋1, 𝕋1dag = f_normality(τ, O)
    ℙinv2 = genP(-τ*2, σx)[2]
    ψ1 = InfiniteMPS([ℂ^2], [ℂ^1])

    ψs, fs, vars = typeof(ψ1)[], Float64[], Float64[]

    for ix in 1:500
        ψ1 = 𝕋1 * ψ1
        a = SvdCut(truncerr(err))
        ψ1 = changebonds(ψ1, a) 
        ψ2 = ℙinv2 * ψ1 
        f = real(log(dot(ψ2, 𝕋1, ψ1) / dot(ψ2, ψ1)))
        var = log(norm(dot(ψ1, 𝕋1dag*𝕋1, ψ1) / dot(ψ1, 𝕋1dag, ψ1) / dot(ψ1, 𝕋1, ψ1)))
        push!(ψs, ψ1)
        push!(fs, f)
        push!(vars, var)
        printstyled("$(left_virtualspace(ψ1, 1)), $(ix), $(var) \n"; color=:red)
    end
    return ψs, fs, vars
end

ψ1s, f1s, var1s = iTEBD_history(0.1, σx, 1e-6)
@save "square_ising/data/badly_gauged-ITEBD-histories_1.jld2" ψ1s f1s var1s  
ψ5s, f5s, var5s = iTEBD_history(0.5, σx, 1e-6)
@save "square_ising/data/badly_gauged-ITEBD-histories_5.jld2" ψ5s f5s var5s  
ψ10s, f10s, var10s = iTEBD_history(1.0, σx, 1e-6)
@save "square_ising/data/badly_gauged-ITEBD-histories_10.jld2" ψ10s f10s var10s  
ψ15s, f15s, var15s = iTEBD_history(1.5, σx, 1e-6)
@save "square_ising/data/badly_gauged-ITEBD-histories_15.jld2" ψ15s f15s var15s  
ψ20s, f20s, var20s = iTEBD_history(2.0, σx, 1e-6)
@save "square_ising/data/badly_gauged-ITEBD-histories_20.jld2" ψ20s f20s var20s  

ψ1s, f1s, var1s = iTEBD_history(0.1, σx, 1e-8)
@save "square_ising/data/badly_gauged-ITEBD_lv2-histories_1.jld2" ψ1s f1s var1s  
ψ5s, f5s, var5s = iTEBD_history(0.5, σx, 1e-8)
@save "square_ising/data/badly_gauged-ITEBD_lv2-histories_5.jld2" ψ5s f5s var5s  
ψ10s, f10s, var10s = iTEBD_history(1.0, σx, 1e-8)
@save "square_ising/data/badly_gauged-ITEBD_lv2-histories_10.jld2" ψ10s f10s var10s  
ψ15s, f15s, var15s = iTEBD_history(1.5, σx, 1e-8)
@save "square_ising/data/badly_gauged-ITEBD_lv2-histories_15.jld2" ψ15s f15s var15s  
ψ20s, f20s, var20s = iTEBD_history(2.0, σx, 1e-8)
@save "square_ising/data/badly_gauged-ITEBD_lv2-histories_20.jld2" ψ20s f20s var20s 

@load "square_ising/data/badly_gauged-ITEBD-histories_1.jld2" ψ1s f1s var1s  
@load "square_ising/data/badly_gauged-ITEBD-histories_5.jld2" ψ5s f5s var5s  
@load "square_ising/data/badly_gauged-ITEBD-histories_10.jld2" ψ10s f10s var10s  
@load "square_ising/data/badly_gauged-ITEBD-histories_15.jld2" ψ15s f15s var15s  
@load "square_ising/data/badly_gauged-ITEBD-histories_20.jld2" ψ20s f20s var20s

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:500, abs.(f1s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:500, abs.(f5s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:500, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:500, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax1, 1:500, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax1)
@show fig 