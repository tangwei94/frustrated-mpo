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

f_normality(0.5, σz)
f_normality(0.5, σx)

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

iTEBD_results_025 = iTEBD_history(0.25, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_025.jld2" iTEBD_results=iTEBD_results_025 

iTEBD_results_050 = iTEBD_history(0.5, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_050.jld2" iTEBD_results=iTEBD_results_050 

iTEBD_results_075 = iTEBD_history(0.75, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_075.jld2" iTEBD_results=iTEBD_results_075 

iTEBD_results_100 = iTEBD_history(1.0, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_100.jld2" iTEBD_results=iTEBD_results_100 

iTEBD_results_125 = iTEBD_history(1.25, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_125.jld2" iTEBD_results=iTEBD_results_125 

iTEBD_results_150 = iTEBD_history(1.5, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_150.jld2" iTEBD_results=iTEBD_results_150 

iTEBD_results_175 = iTEBD_history(1.75, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_175.jld2" iTEBD_results=iTEBD_results_175 

iTEBD_results_200 = iTEBD_history(2.0, σx, 1e-8);
@save "square_ising/data/badly_gauged-ITEBD-histories_200.jld2" iTEBD_results=iTEBD_results_200 

iTEBD_results_vec = map(indices) do index
    @load "square_ising/data/badly_gauged-ITEBD-histories_$(index).jld2" iTEBD_results
    return iTEBD_results
end

indices = ["025", "050", "075", "100", "125", "150", "175", "200"];
τs = 0.25:0.25:2.00;
ferrs = map(zip(iTEBD_results_vec, τs)) do item 
    iTEBD_results, τ = item 
    
    ψ = iTEBD_results[1][end]
    ℙ = genP(τ, σx)[2]
    ℙinv = genP(-τ, σx)[2]

    ψ1 = ℙinv * ψ
    f = real(log(dot(ψ1, 𝕋0, ψ1) / dot(ψ1, ψ1)))
    return abs.(f .- f_exact) / f_exact
end
vars = map(iTEBD_results_vec) do iTEBD_results 
    iTEBD_results[4][end]
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{error in }f", yscale=log10)
scatter!(ax1, τs, ferrs, marker=:circle, markersize=10)
ax2 = Axis(fig[1, 2], xlabel=L"\tau", ylabel=L"\text{variance}", yscale=log10)
scatter!(τs, norm.(vars) .+ 1e-16, marker=:circle, markersize=10)
save("square_ising/data/fig-badly_gauged-iTEBD-sx.pdf", fig)
@show fig

# plot f_err vs steps 
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 900))
ax1 = Axis(fig[1:2, 1], ylabel=L"\text{error in }f", xticklabelsvisible=false, yscale=log10)
for (iTEBD_results, τ) in zip(iTEBD_results_vec, τs)
    lines!(ax1, 1:500, abs.(iTEBD_results[3] .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
end
#axislegend(ax1)
@show fig 

ax2 = Axis(fig[3:4, 1], ylabel=L"\text{bond dimension}", xticklabelsvisible=false) 
for (iTEBD_results, τ) in zip(iTEBD_results_vec, τs)
    χs = map(iTEBD_results[1]) do ψR 
        return dim(left_virtualspace(ψR, 1))
    end
    lines!(ax2, 1:500, χs, label=latexstring("\$τ=$(τ)\$"))
end
@show fig

ax3 = Axis(fig[5:6, 1], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10) 
for (iTEBD_results, τ) in zip(iTEBD_results_vec, τs)
    lines!(ax3, 1:500, norm.(iTEBD_results[4]) .+ 1e-16, label=latexstring("\$τ=$(τ)\$"))
end
Legend(fig[7, 1], ax1, nbanks=4)
@show fig
save("square_ising/data/badly_gauged-iTEBD-histories.pdf", fig)

####### to be updated. z



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