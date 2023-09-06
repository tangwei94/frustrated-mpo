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

Ns = map(-1.0:0.1:0.1) do τ
    _, 𝕋1, _ = f_normality(τ, σy)
    norm(𝕋1.opp[1])
end

function iTEBD_history(τ::Real, O::AbstractTensorMap, err::Real)
    _, 𝕋1, 𝕋1dag = f_normality(τ, O)
    ψR = InfiniteMPS([ℂ^2], [ℂ^1])
    ψL = InfiniteMPS([ℂ^2], [ℂ^1])

    ψRs, ψLs, fs, vars = typeof(ψR)[], typeof(ψL)[], Float64[], Float64[]
    alg = SvdCut(truncerr(err))

    for ix in 1:500
        ψR = changebonds(𝕋1 * ψR, alg)
        ψL = changebonds(𝕋1dag * ψL, alg)
        
        f = real(log(dot(ψL, 𝕋1, ψR) / dot(ψL, ψR)))
        varR = log(norm(dot(ψR, 𝕋1dag*𝕋1, ψR) / dot(ψR, 𝕋1dag, ψR) / dot(ψR, 𝕋1, ψR)))
        push!(ψRs, ψR)
        push!(ψLs, ψL)
        push!(fs, f)
        push!(vars, varR)
        printstyled("$(left_virtualspace(ψR, 1)), $(ix), $(varR) \n"; color=:red)
    end
    return ψRs, ψLs, fs, vars
end

iTEBD_results_01 = iTEBD_history(0.1, σx, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_01.jld2" iTEBD_results=iTEBD_results_01
f01s = iTEBD_results_01[3];

iTEBD_results_05 = iTEBD_history(0.5, σx, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_05.jld2" iTEBD_results=iTEBD_results_05 
f05s = iTEBD_results_05[3]; 

iTEBD_results_10 = iTEBD_history(1.0, σx, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_10.jld2" iTEBD_results=iTEBD_results_10 
f10s = iTEBD_results_10[3]; 

iTEBD_results_15 = iTEBD_history(1.5, σx, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_15.jld2" iTEBD_results=iTEBD_results_15 
f15s = iTEBD_results_15[3]; 

iTEBD_results_20 = iTEBD_history(2.0, σx, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_20.jld2" iTEBD_results=iTEBD_results_20 
f20s = iTEBD_results_20[3]; 

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:500, abs.(f01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:500, abs.(f05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:500, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:500, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax1, 1:500, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{bond dimension}") 
get_bondD(res) = map(res[1]) do ψR 
    return dim(left_virtualspace(ψR, 1))
end

χs_01 = get_bondD(iTEBD_results_01);
χs_05 = get_bondD(iTEBD_results_05);
χs_10 = get_bondD(iTEBD_results_10);
χs_15 = get_bondD(iTEBD_results_15);
χs_20 = get_bondD(iTEBD_results_20);

lines!(ax2, 1:500, χs_01, label=L"τ=0.1")
lines!(ax2, 1:500, χs_05, label=L"τ=0.5")
lines!(ax2, 1:500, χs_10, label=L"τ=1.0")
lines!(ax2, 1:500, χs_15, label=L"τ=1.5")
lines!(ax2, 1:500, χs_20, label=L"τ=2.0")

axislegend(ax2)

@show fig

ax3 = Axis(fig[3, 1], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10) 
get_var(res) = res[4]

res_01 = get_var(iTEBD_results_01);
res_05 = get_var(iTEBD_results_05);
res_10 = get_var(iTEBD_results_10);
res_15 = get_var(iTEBD_results_15);
res_20 = get_var(iTEBD_results_20);

lines!(ax3, 1:500, abs.(res_01), label=L"τ=0.1")
lines!(ax3, 1:500, abs.(res_05), label=L"τ=0.5")
lines!(ax3, 1:500, abs.(res_10), label=L"τ=1.0")
lines!(ax3, 1:500, abs.(res_15), label=L"τ=1.5")
lines!(ax3, 1:500, abs.(res_20), label=L"τ=2.0")

axislegend(ax3)

@show fig
save("square_ising/data/badly_gauged-iTEBD-histories.pdf", fig)

iTEBD_results_01 = iTEBD_history(0.1, σz, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_z_01.jld2" iTEBD_results=iTEBD_results_01
f01s = iTEBD_results_01[3];

iTEBD_results_05 = iTEBD_history(0.5, σz, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_z_05.jld2" iTEBD_results=iTEBD_results_05 
f05s = iTEBD_results_05[3]; 

iTEBD_results_10 = iTEBD_history(1.0, σz, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_z_10.jld2" iTEBD_results=iTEBD_results_10 
f10s = iTEBD_results_10[3]; 

iTEBD_results_15 = iTEBD_history(1.5, σz, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_z_15.jld2" iTEBD_results=iTEBD_results_15 
f15s = iTEBD_results_15[3]; 

iTEBD_results_20 = iTEBD_history(2.0, σz, 1e-6);
@save "square_ising/data/badly_gauged-ITEBD-histories_z_20.jld2" iTEBD_results=iTEBD_results_20 
f20s = iTEBD_results_20[3]; 

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
lines!(ax1, 1:500, abs.(f01s .- f_exact) ./ f_exact, label=L"τ=0.1")
lines!(ax1, 1:500, abs.(f05s .- f_exact) ./ f_exact, label=L"τ=0.5")
lines!(ax1, 1:500, abs.(f10s .- f_exact) ./ f_exact, label=L"τ=1.0")
lines!(ax1, 1:500, abs.(f15s .- f_exact) ./ f_exact, label=L"τ=1.5")
lines!(ax1, 1:500, abs.(f20s .- f_exact) ./ f_exact, label=L"τ=2.0")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{bond dimension}") 
get_bondD(res) = map(res[1]) do ψR 
    return dim(left_virtualspace(ψR, 1))
end

χs_01 = get_bondD(iTEBD_results_01);
χs_05 = get_bondD(iTEBD_results_05);
χs_10 = get_bondD(iTEBD_results_10);
χs_15 = get_bondD(iTEBD_results_15);
χs_20 = get_bondD(iTEBD_results_20);

lines!(ax2, 1:500, χs_01, label=L"τ=0.1")
lines!(ax2, 1:500, χs_05, label=L"τ=0.5")
lines!(ax2, 1:500, χs_10, label=L"τ=1.0")
lines!(ax2, 1:500, χs_15, label=L"τ=1.5")
lines!(ax2, 1:500, χs_20, label=L"τ=2.0")

axislegend(ax2)

@show fig

ax3 = Axis(fig[3, 1], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10) 
get_var(res) = res[4]

res_01 = get_var(iTEBD_results_01);
res_05 = get_var(iTEBD_results_05);
res_10 = get_var(iTEBD_results_10);
res_15 = get_var(iTEBD_results_15);
res_20 = get_var(iTEBD_results_20);

lines!(ax3, 1:500, abs.(res_01), label=L"τ=0.1")
lines!(ax3, 1:500, abs.(res_05), label=L"τ=0.5")
lines!(ax3, 1:500, abs.(res_10), label=L"τ=1.0")
lines!(ax3, 1:500, abs.(res_15), label=L"τ=1.5")
lines!(ax3, 1:500, abs.(res_20), label=L"τ=2.0")

axislegend(ax3)

@show fig
save("square_ising/data/badly_gauged-iTEBD-histories_z.pdf", fig)