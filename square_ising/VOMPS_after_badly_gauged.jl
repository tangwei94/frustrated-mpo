using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie
using QuadGK 

include("../utils.jl");

βc = asinh(1) / 2
k = 1 / (sinh(2*βc))^2
f_exact = log(2) / 2 + (1/2/pi) * quadgk(θ-> log(cosh(2*βc)*cosh(2*βc) + (1/k)*sqrt(1+k^2-2*k*cos(2*θ))), 0, pi, rtol = 1e-12)[1]

T = tensor_square_ising(βc)

σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im; -im 0], ℂ^2, ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
function genP(τ::Real, O::AbstractTensorMap)
    P = add_util_leg(exp(-τ*O))
    ℙ = DenseMPO([P])
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

indices = ["000", "025", "050", "075", "100", "125", "150", "175"];
τs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
χs = [2, 4, 8, 16, 32, 64]

index = "000"
VOMPS_results = VOMPS_history(0, σz);
@save "square_ising/data/badly_gauged-VOMPS-histories_z_$(index).jld2" VOMPS_results 
@save "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results 

for (index, τ) in zip(indices[1:end], τs[1:end]) 
    VOMPS_results = VOMPS_history(τ, σx);
    @save "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results 
end

VOMPS_results_vec = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    return VOMPS_results
end; 

ferrs = map(zip(VOMPS_results_vec, τs)) do item 
    VOMPS_results, τ = item 
    
    ψ = VOMPS_results[1][end]
    ℙ = genP(τ, σx)[2]
    ℙinv = genP(-τ, σx)[2]

    ψ1 = ℙinv * ψ
    f = real(log(dot(ψ1, 𝕋0, ψ1) / dot(ψ1, ψ1)))
    return abs.(f .- f_exact) / f_exact
end
vars = map(VOMPS_results_vec) do VOMPS_results 
    VOMPS_results[4][end]
end
iTEBD_results_vec = map(indices) do index
    @load "square_ising/data/badly_gauged-ITEBD-histories_$(index).jld2" iTEBD_results
    return iTEBD_results
end
ferrs_iTEBD = map(zip(iTEBD_results_vec, τs)) do item 
    iTEBD_results, τ = item 
    
    ψ = iTEBD_results[1][end]
    ℙ = genP(τ, σx)[2]
    ℙinv = genP(-τ, σx)[2]

    ψ1 = ℙinv * ψ
    f = real(log(dot(ψ1, 𝕋0, ψ1) / dot(ψ1, ψ1)))
    return abs.(f .- f_exact) / f_exact
end
vars_iTEBD = map(iTEBD_results_vec) do iTEBD_results 
    iTEBD_results[4][end]
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{error in }f", yscale=log10)
scatter!(ax1, τs, ferrs_iTEBD, marker=:circle, markersize=10, label=L"\text{iTEBD}")
lines!(ax1, τs, ferrs_iTEBD, linestyle=:dash, label=L"\text{iTEBD}")
scatter!(ax1, τs, ferrs, marker=:circle, markersize=10, label=L"\text{VOMPS}")
lines!(ax1, τs, ferrs, linestyle=:dash, label=L"\text{VOMPS}")
axislegend(ax1, position=:lt, merge=true)
ax2 = Axis(fig[1, 2], xlabel=L"\tau", ylabel=L"\text{variance}", yscale=log10)
scatter!(τs, norm.(vars_iTEBD), marker=:circle, markersize=10, label=L"\text{iTEBD}")
lines!(τs, norm.(vars_iTEBD), linestyle=:dash, label=L"\text{iTEBD}")
scatter!(τs, norm.(vars), marker=:circle, markersize=10, label=L"\text{VOMPS}")
lines!(τs, norm.(vars), linestyle=:dash, label=L"\text{VOMPS}")
axislegend(ax2, position=:rt, merge=true)
save("square_ising/data/fig-badly_gauged-VOMPS-sx.pdf", fig)
@show fig

# detailed histories
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 700))
ax1 = Axis(fig[1:3, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
for (index, τ) in zip(index_arr, τs) 
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    fs = VOMPS_results[3]
    lines!(ax1, 1:1500, abs.(fs .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
end

ax2 = Axis(fig[4:6, 1], xlabel=L"χ", ylabel=L"\text{error in }f", yscale=log10)
χs = 2 .^ (1:6)
for (index, τ) in zip(index_arr, τs) 
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    f_res = VOMPS_results[3][250:250:end]
    lines!(ax2, χs, abs.(f_res .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
    scatter!(ax2, χs, abs.(f_res .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
end
Legend(fig[end+1, 1], ax1, nbanks=5)
save("square_ising/data/badly_gauged-VOMPS-histories.pdf", fig)
@show fig

# sigma z results
indices = ["000", "025", "050", "075", "100", "125", "150", "175"];
τs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
χs = [2, 4, 8, 16, 32, 64]

for (index, τ) in zip(indices[1:end], τs[1:end]) 
    VOMPS_results = VOMPS_history(τ, σz);
    @save "square_ising/data/badly_gauged-VOMPS-histories_z_$(index).jld2" VOMPS_results 
end

VOMPS_results_vec = map(indices) do index
    @load "square_ising/data/badly_gauged-VOMPS-histories_z_$(index).jld2" VOMPS_results
    return VOMPS_results
end; 

ferrs = map(zip(VOMPS_results_vec, τs)) do item 
    VOMPS_results, τ = item 
    
    ψ = VOMPS_results[1][end]
    ℙ = genP(τ, σz)[2]
    ℙinv = genP(-τ, σz)[2]

    ψ1 = ℙinv * ψ
    f = real(log(dot(ψ1, 𝕋0, ψ1) / dot(ψ1, ψ1)))
    return abs.(f .- f_exact) / f_exact
end
vars = map(VOMPS_results_vec) do VOMPS_results 
    VOMPS_results[4][end]
end
iTEBD_results_vec = map(indices) do index
    @load "square_ising/data/badly_gauged-ITEBD-histories_z_$(index).jld2" iTEBD_results
    return iTEBD_results
end
ferrs_iTEBD = map(zip(iTEBD_results_vec, τs)) do item 
    iTEBD_results, τ = item 
    
    ψ = iTEBD_results[1][end]
    ℙ = genP(τ, σz)[2]
    ℙinv = genP(-τ, σz)[2]

    ψ1 = ℙinv * ψ
    f = real(log(dot(ψ1, 𝕋0, ψ1) / dot(ψ1, ψ1)))
    return abs.(f .- f_exact) / f_exact
end
vars_iTEBD = map(iTEBD_results_vec) do iTEBD_results 
    iTEBD_results[4][end]
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{error in }f", yscale=log10)
scatter!(ax1, τs, ferrs_iTEBD, marker=:circle, markersize=10, label=L"\text{iTEBD}")
lines!(ax1, τs, ferrs_iTEBD, linestyle=:dash, label=L"\text{iTEBD}")
scatter!(ax1, τs, ferrs, marker=:circle, markersize=10, label=L"\text{VOMPS}")
lines!(ax1, τs, ferrs, linestyle=:dash, label=L"\text{VOMPS}")
axislegend(ax1, position=:lt, merge=true)
ax2 = Axis(fig[1, 2], xlabel=L"\tau", ylabel=L"\text{variance}", yscale=log10)
scatter!(τs, norm.(vars_iTEBD), marker=:circle, markersize=10, label=L"\text{iTEBD}")
lines!(τs, norm.(vars_iTEBD), linestyle=:dash, label=L"\text{iTEBD}")
scatter!(τs, norm.(vars), marker=:circle, markersize=10, label=L"\text{VOMPS}")
lines!(τs, norm.(vars), linestyle=:dash, label=L"\text{VOMPS}")
axislegend(ax2, position=:rt, merge=true)
save("square_ising/data/fig-badly_gauged-VOMPS-sz.pdf", fig)
@show fig

# detailed histories
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 700))
ax1 = Axis(fig[1:3, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
for (index, τ) in zip(indices, τs) 
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    fs = VOMPS_results[3]
    lines!(ax1, 1:1500, abs.(fs .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
end

ax2 = Axis(fig[4:6, 1], xlabel=L"χ", ylabel=L"\text{error in }f", yscale=log10)
χs = 2 .^ (1:6)
for (index, τ) in zip(indices, τs) 
    @load "square_ising/data/badly_gauged-VOMPS-histories_$(index).jld2" VOMPS_results
    f_res = VOMPS_results[3][250:250:end]
    lines!(ax2, χs, abs.(f_res .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
    scatter!(ax2, χs, abs.(f_res .- f_exact) ./ f_exact, label=latexstring("\$τ=$(τ)\$"))
end
Legend(fig[end+1, 1], ax1, nbanks=5)
save("square_ising/data/badly_gauged-VOMPS-histories_z.pdf", fig)
@show fig

