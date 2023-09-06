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

M = σz ⊗ σz
function genPmpo(τ::Real)
    L, S, R = tsvd(exp(-τ * M), (1, 3), (2, 4), trunc=truncerr(1e-10))
    L = permute(L * sqrt(S), (1, ), (2, 3))
    R = permute(sqrt(S) * R, (1, 2), (3, ))
    @tensor T1[-1 -2; -3 -4] := L[-2; 1 -4] * R[-1 1 ; -3]
    @tensor T2[-1 -2; -3 -4] := R[-1 -2; 1] * L[1; -3 -4]
    #@show norm(T1 - T2)
    return DenseMPO([T1])
end

𝕋0 = mpo_gen(1, T, :inf)

function mpo_ovlp(A1, A2)
    χ1 = dim(MPSKit._lastspace(A1))
    χ2 = dim(MPSKit._lastspace(A2))

    function mpo_transf(v)
        @tensor Tv[-1; -2] := A1[-1 3; 4 1] * conj(A2[-2 3; 4 2]) * v[1; 2]
        return Tv
    end

    v0 = TensorMap(rand, ComplexF64, ℂ^χ1, ℂ^χ2)
    return eigsolve(mpo_transf, v0, 1, :LM)
end

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

function mpof_normality(τ::Real)
    ℙ = genPmpo(τ)
    ℙinv = genPmpo(-τ)

    𝕋1 = ℙ * 𝕋0 * ℙinv
    𝕋1dag = ℙinv * 𝕋0 * ℙ 

    a1 = 𝕋1.opp[1]
    a2 = 𝕋1dag.opp[1]

    normality = real(mpo_ovlp(a1, a2)[1][1] * mpo_ovlp(a2, a1)[1][1] / mpo_ovlp(a1, a1)[1][1] / mpo_ovlp(a2, a2)[1][1])

    return normality, 𝕋1, 𝕋1dag
end

function VOMPS_history(τ::Real, O::AbstractTensorMap)
    _, 𝕋1, 𝕋1dag = f_normality(τ, O)
    optim_alg1 = VUMPS(tol_galerkin=1e-9, maxiter=100) 
    ψR = InfiniteMPS([ℂ^2], [ℂ^1])
    ψL = InfiniteMPS([ℂ^2], [ℂ^1])

    ψRs, ψLs, fs, vars = typeof(ψR)[], typeof(ψL)[], Float64[], Float64[]

    for _ in 1:6
        ψR = 𝕋1 * ψR
        ψL = 𝕋1dag * ψL
        for ix in 1:250
            ψR, _ = approximate(ψR, (𝕋1, ψR), optim_alg1)
            ψL, _ = approximate(ψL, (𝕋1dag, ψL), optim_alg1)
            f = real(log(dot(ψL, 𝕋1, ψR) / dot(ψL, ψR)))
            var = log(norm(dot(ψR, 𝕋1dag*𝕋1, ψR) / dot(ψR, 𝕋1dag, ψR) / dot(ψR, 𝕋1, ψR)))
            push!(ψRs, ψR)
            push!(ψLs, ψL)
            push!(fs, f)
            push!(vars, var)
            printstyled("$(left_virtualspace(ψR, 1)), $(ix), $(var) \n"; color=:red)
        end
    end
    return ψRs, ψLs, fs, vars
end

function VOMPS_history(𝕋1::DenseMPO, 𝕋1dag::DenseMPO)
    optim_alg1 = VUMPS(tol_galerkin=1e-9, maxiter=100) 
    ψR = InfiniteMPS([ℂ^2], [ℂ^1])
    ψL = InfiniteMPS([ℂ^2], [ℂ^1])

    ψRs, ψLs, fs, vars = typeof(ψR)[], typeof(ψL)[], Float64[], Float64[]

    for _ in 1:2
        ψR = 𝕋1 * ψR
        ψL = 𝕋1dag * ψL
        for ix in 1:250
            ψR, _ = approximate(ψR, (𝕋1, ψR), optim_alg1)
            ψL, _ = approximate(ψL, (𝕋1dag, ψL), optim_alg1)
            f = real(log(dot(ψL, 𝕋1, ψR) / dot(ψL, ψR)))
            var = log(norm(dot(ψR, 𝕋1dag*𝕋1, ψR) / dot(ψR, 𝕋1dag, ψR) / dot(ψR, 𝕋1, ψR)))
            push!(ψRs, ψR)
            push!(ψLs, ψL)
            push!(fs, f)
            push!(vars, var)
            printstyled("$(left_virtualspace(ψR, 1)), $(ix), $(var) \n"; color=:red)
        end
    end
    return ψRs, ψLs, fs, vars
end


VOMPS_results_01 = VOMPS_history(0.1, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_01.jld2" VOMPS_results=VOMPS_results_01 
f01s = VOMPS_results_01[3];

VOMPS_results_05 = VOMPS_history(0.5, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_05.jld2" VOMPS_results=VOMPS_results_05 
f05s = VOMPS_results_05[3];

VOMPS_results_10 = VOMPS_history(1.0, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_10.jld2" VOMPS_results=VOMPS_results_10 
f10s = VOMPS_results_10[3];

VOMPS_results_15 = VOMPS_history(1.5, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_15.jld2" VOMPS_results=VOMPS_results_15 
f15s = VOMPS_results_15[3];

VOMPS_results_20 = VOMPS_history(2.0, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_20.jld2" VOMPS_results=VOMPS_results_20 
f20s = VOMPS_results_20[3];

VOMPS_results_30 = VOMPS_history(3.0, σx);
@save "square_ising/data/badly_gauged-VOMPS-histories_30.jld2" VOMPS_results=VOMPS_results_30 
f30s = VOMPS_results_30[3];

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:1500, abs.(f01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:1500, abs.(f05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:1500, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:1500, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax1, 1:1500, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
lines!(ax1, 1:1500, abs.(f30s .- f_exact) ./ f_exact, label=L"τ=3.0")
axislegend(ax1)
@show fig 

get_results(res) = res[3][250:250:end]
χs = 2 .^ (1:6)
f_res01s = get_results(VOMPS_results_01)
f_res05s = get_results(VOMPS_results_05)
f_res10s = get_results(VOMPS_results_10)
f_res15s = get_results(VOMPS_results_15)
f_res20s = get_results(VOMPS_results_20)

ax2 = Axis(fig[2, 1], xlabel=L"χ", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax2, χs, abs.(f_res01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax2, χs, abs.(f_res05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax2, χs, abs.(f_res10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax2, χs, abs.(f_res15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax2, χs, abs.(f_res20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax2)
@show fig

save("square_ising/data/badly_gauged-VOMPS-histories.pdf", fig)

VOMPS_results_01 = VOMPS_history(0.1, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_01.jld2" VOMPS_results=VOMPS_results_01 
f01s = VOMPS_results_01[3];

VOMPS_results_05 = VOMPS_history(0.5, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_05.jld2" VOMPS_results=VOMPS_results_05 
f05s = VOMPS_results_05[3];

VOMPS_results_10 = VOMPS_history(1.0, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_10.jld2" VOMPS_results=VOMPS_results_10 
f10s = VOMPS_results_10[3];

VOMPS_results_15 = VOMPS_history(1.5, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_15.jld2" VOMPS_results=VOMPS_results_15 
f15s = VOMPS_results_15[3];

VOMPS_results_20 = VOMPS_history(2.0, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_20.jld2" VOMPS_results=VOMPS_results_20 
f20s = VOMPS_results_20[3];

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:1500, abs.(f01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:1500, abs.(f05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:1500, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:1500, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax1, 1:1500, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax1)
@show fig 

get_results(res) = res[3][250:250:end]
χs = 2 .^ (1:6)
f_res01s = get_results(VOMPS_results_01)
f_res05s = get_results(VOMPS_results_05)
f_res10s = get_results(VOMPS_results_10)
f_res15s = get_results(VOMPS_results_15)
f_res20s = get_results(VOMPS_results_20)

ax2 = Axis(fig[2, 1], xlabel=L"χ", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax2, χs, abs.(f_res01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax2, χs, abs.(f_res05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax2, χs, abs.(f_res10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax2, χs, abs.(f_res15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax2, χs, abs.(f_res20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax2)
@show fig

save("square_ising/data/badly_gauged-VOMPS-histories_z.pdf", fig)

