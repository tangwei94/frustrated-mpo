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

function VOMPS_history(τ::Real, O::AbstractTensorMap)
    normality1, 𝕋1, 𝕋1dag = f_normality(τ, O)
    optim_alg1 = VUMPS(tol_galerkin=1e-9, maxiter=100) 
    ℙinv2 = genP(-τ*2, σx)[2]
    ψ1 = InfiniteMPS([ℂ^2], [ℂ^1])

    ψs, fs, vars = typeof(ψ1)[], Float64[], Float64[]

    for _ in 1:6
        ψ1 = 𝕋1 * ψ1
        for ix in 1:500
            ψ1, _ = approximate(ψ1, (𝕋1, ψ1), optim_alg1)
            ψ2 = ℙinv2 * ψ1 
            f = real(log(dot(ψ2, 𝕋1, ψ1) / dot(ψ2, ψ1)))
            var = log(norm(dot(ψ1, 𝕋1dag*𝕋1, ψ1) / dot(ψ1, 𝕋1dag, ψ1) / dot(ψ1, 𝕋1, ψ1)))
            push!(ψs, ψ1)
            push!(fs, f)
            push!(vars, var)
            printstyled("$(left_virtualspace(ψ1, 1)), $(ix), $(var) \n"; color=:red)
        end
    end
    return ψs, fs, vars
end
function VOMPS_history_bi(τ::Real, O::AbstractTensorMap)
    normality1, 𝕋1, 𝕋1dag = f_normality(τ, O)
    optim_alg1 = VUMPS(tol_galerkin=1e-9, maxiter=100) 
    ψ1 = InfiniteMPS([ℂ^2], [ℂ^1])
    ψ2 = InfiniteMPS([ℂ^2], [ℂ^1])

    ψs, fs, vars = [], Float64[], Float64[]

    for _ in 1:6
        ψ1 = 𝕋1 * ψ1
        ψ2 = 𝕋1dag * ψ2
        for ix in 1:500
            ψ1, _ = approximate(ψ1, (𝕋1, ψ1), optim_alg1)
            ψ2, _ = approximate(ψ2, (𝕋1dag, ψ2), optim_alg1)
            f = real(log(dot(ψ2, 𝕋1, ψ1) / dot(ψ2, ψ1)))
            var = log(norm(dot(ψ1, 𝕋1dag*𝕋1, ψ1) / dot(ψ1, 𝕋1dag, ψ1) / dot(ψ1, 𝕋1, ψ1)))
            push!(ψs, (ψ1, ψ2))
            push!(fs, f)
            push!(vars, var)
            printstyled("$(left_virtualspace(ψ1, 1)), $(ix), $(var) \n"; color=:red)
        end
    end
    return ψs, fs, vars
end

ψ1s, f1s, var1s =  VOMPS_history(0.1, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_1.jld2" ψ1s f1s var1s  
ψ5s, f5s, var5s =  VOMPS_history(0.5, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_5.jld2" ψ5s f5s var5s  
ψ10s, f10s, var10s =  VOMPS_history(1.0, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_10.jld2" ψ10s f10s var10s  
ψ15s, f15s, var15s =  VOMPS_history(1.5, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_15.jld2" ψ15s f15s var15s  
ψ20s, f20s, var20s =  VOMPS_history(2.0, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_20.jld2" ψ20s f20s var20s  

@load "square_ising/data/badly_gauged-VOMPS-histories_1.jld2" ψ1s f1s var1s  
@load "square_ising/data/badly_gauged-VOMPS-histories_5.jld2" ψ5s f5s var5s  
@load "square_ising/data/badly_gauged-VOMPS-histories_10.jld2" ψ10s f10s var10s  
@load "square_ising/data/badly_gauged-VOMPS-histories_15.jld2" ψ15s f15s var15s  
@load "square_ising/data/badly_gauged-VOMPS-histories_20.jld2" ψ20s f20s var20s  

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:3000, abs.(f1s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:3000, abs.(f5s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:3000, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:3000, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
#lines!(ax1, 1:3000, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"τ", ylabel=L"\text{variance}", yscale=log10)
lines!(ax2, 1:3000, abs.(var1s) .+ 1e-16, label=L"τ=0.1")
lines!(ax2, 1:3000, abs.(var5s) .+ 1e-16, label=L"τ=0.5")
lines!(ax2, 1:3000, abs.(var10s) .+ 1e-16, label=L"τ=1.0")
lines!(ax2, 1:3000, abs.(var15s) .+ 1e-16, label=L"τ=1.5")
lines!(ax2, 1:3000, abs.(var20s) .+ 1e-16, label=L"τ=2.0")
axislegend(ax2)
@show fig
save("square_ising/data/badly_gauged-VOMPS-histories.pdf", fig)
# not easy to see the difference. change strategy