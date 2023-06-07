using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
M = σz ⊗ σz
function genP(τ::Real)
    L, S, R = tsvd(exp(-τ * M), (1, 3), (2, 4), trunc=truncerr(1e-10))
    L = permute(L * sqrt(S), (1, ), (2, 3))
    R = permute(sqrt(S) * R, (1, 2), (3, ))
    @tensor T1[-1 -2; -3 -4] := L[-2; 1 -4] * R[-1 1 ; -3]
    @tensor T2[-1 -2; -3 -4] := R[-1 -2; 1] * L[1; -3 -4]
    #@show norm(T1 - T2)
    return T1
end

function f_normality(τ::Real)
    if τ === Inf
        @show "inf"
        Tn1 = tensor_triangular_AF_ising_adapted()
        Tndag1 = mpotensor_dag(Tn1)
        χ = 4
        @show Tn1
    else 
        ℙ = mpo_gen(1, genP(τ), :Inf)
        ℙinv = mpo_gen(1, genP(-τ), :Inf)

        𝕋n = ℙ * 𝕋 * ℙinv
        𝕋ndag = ℙinv * 𝕋dag * ℙ 

        # original MPO has redundancies. conversion to iMPS will fail to converge when MPSKit tries to find canonical forms
        Tn = 𝕋n.opp[1]
        L, S, R = tsvd(Tn, (1,2,3), (4,), trunc=truncerr(1e-10))
        @tensor Tn1[-1 -2 ; -3 -4] := S[-1; 1] * R[1; 2] * L[2 -2; -3 -4] 
        χ = dim(domain(S))
        @show χ
        Tndag = 𝕋ndag.opp[1]
        L, S, R = tsvd(Tndag, (1,2,3), (4,), trunc=truncerr(1e-10))
        @tensor Tndag1[-1 -2 ; -3 -4] := S[-1; 1] * R[1; 2] * L[2 -2; -3 -4] 
    end

    function f_normality2(K::AbstractTensorMap)
        K = 0.5*(K+K')
        G = exp(K)
        Ginv = exp(-1*K)
        @tensor Tnorm = G[1; 2] * Tn1[2 5; 6 3] * Ginv[3; 4] * (Tn1')[6 4; 1 5]
        return sqrt(norm(Tnorm))
    end
    _fg(K) = (f_normality2(K), f_normality2'(K))
    inner(K, K1, K2) = real(dot(K1, K2))

    optalg_LBFGS = OptimKit.LBFGS(;maxiter=200, gradtol=1e-12, verbosity=1)
    
    K0 = TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ)
    Kopt, fvalue, _, _, _ = OptimKit.optimize(_fg, K0, optalg_LBFGS; inner=inner)
    @show fvalue
    G = exp(0.5*(Kopt+Kopt'))
    Λ, U = eigh(G)
    P = sqrt(Λ) * U'
    Pinv = U * sqrt(inv(Λ))
    @tensor Tn1_n[-1 -2; -3 -4] := P[-1; 1] * Tn1[1 -2; -3 2] * Pinv[2; -4]

    𝕋n1 = mpo_gen(1, Tn1_n, :inf)
    𝕋ndag1 = mpo_gen(1, mpotensor_dag(Tn1_n), :inf)
    ψ = convert(InfiniteMPS, 𝕋n1*𝕋ndag1)
    ϕ = convert(InfiniteMPS, 𝕋ndag1*𝕋n1)

    return norm(dot(ψ, ϕ)), 𝕋n1

end

normality, 𝕋n1 = f_normality(5)
normality, 𝕋n_nf = f_normality(Inf)

aaa_nf = convert(InfiniteMPS, 𝕋n_nf)
bbb_nf = convert(InfiniteMPS, 𝕋n1)
dot(aaa_nf, bbb_nf)

normalities = Float64[]
fidelities_with_Tnf = Float64[]
τs = -4:0.5:4
for τ in τs 
    y, 𝕋tmp = f_normality(τ)

    aaa_nf = convert(InfiniteMPS, 𝕋n_nf)
    bbb_nf = convert(InfiniteMPS, 𝕋tmp)

    push!(normalities, y)
    push!(fidelities_with_Tnf, norm(dot(aaa_nf, bbb_nf)))
    @show τ, y, log(y)
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\mathrm{fidel}(T T^\dagger, T^\dagger T)")
scatter1 = CairoMakie.scatter!(ax1, τs, normalities, marker=:circle, markersize=10)
@show fig
ax2 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"\mathrm{fidel}(T, T_0)")
scatter2 = CairoMakie.scatter!(ax2, τs, fidelities_with_Tnf, marker=:circle, markersize=10)
@show fig
save("gauged-frustrated-mpo-normal-meas.pdf", fig)

function optimize_with_τ(τ::Real, maxiter::Int=10000)
    if τ === Inf
        normality, Tn1 = 1, tensor_triangular_AF_ising_adapted()
        𝕋n1 = mpo_gen(1, Tn1, :inf)
    else
        normality, 𝕋n1 = f_normality(τ)
    end
    
    ψt0 = InfiniteMPS([ℂ^2], [ℂ^8])

    expand_alg = OptimalExpand(truncdim(8))
    optim_alg = VUMPS(tol_galerkin=1e-12, maxiter=maxiter)
    fs = Float64[]
    ψs = typeof(ψt0)[] 
    for ix in 1:8
        ψt, envt, _ = leading_boundary(ψt0, 𝕋n1, optim_alg)
        ψt0, _ = changebonds(ψt, 𝕋n1, expand_alg, envt)

        f1 = log(norm(dot(ψt, 𝕋n1, ψt)))
        @show space(ψt.AL[1]), f1 
        push!(fs, f1)
        push!(ψs, ψt)
    end
    return ψs, fs, normality
end

ψsinf, fsinf, normalityinf = optimize_with_τ(Inf, 10000)
ψs3, fs3, normality3 = optimize_with_τ(3, 10000)
ψs2, fs2, normality2 = optimize_with_τ(2, 10000)
ψs1, fs1, normality1 = optimize_with_τ(1, 10000)

@save "gauge_AF_triangular_ising/data/VUMPS_data.jld2" ψsinf ψs3 ψs2 ψs1 fsinf fs3 fs2 fs1

f_exact = 0.3230659669

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\chi", ylabel=L"\text{error in }f", yscale=log10)
scatterinf = CairoMakie.scatter!(ax1, 8:8:64, abs.(fsinf .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"\text{good MPO}")
scatter3 = CairoMakie.scatter!(ax1, 8:8:64, abs.(fs3 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"\tau=3")
scatter2 = CairoMakie.scatter!(ax1, 8:8:64, abs.(fs2 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"\tau=2")
scatter1 = CairoMakie.scatter!(ax1, 8:8:64, abs.(fs1 .- f_exact) ./ f_exact, marker=:circle, markersize=10, label=L"\tau=1")
axislegend(ax1)
@show fig 
save("gauge_AF_triangular_ising/data/VUMPS_plot.pdf", fig)


function VUMPS_history(τ::Real, χ::Int, maxiter::Int)
    normality, 𝕋n1 = f_normality(τ)
    @show norm(𝕋n1.opp[1])
    ψt0 = InfiniteMPS([ℂ^2], [ℂ^χ])

    f_history = Float64[]
    galerkin_history = Float64[]
    function finalize1(iter,state,H,envs)
        st = convert(InfiniteMPS,state) 
        f1 = log(norm(dot(st, 𝕋n1, st)))
        push!(f_history, f1)
        push!(galerkin_history, MPSKit.calc_galerkin(state, envs))
        @show length(f_history), length(galerkin_history)
        return (state, envs)
    end

    optim_alg = VUMPS(tol_galerkin=1e-12, maxiter=maxiter, finalize=finalize1)
    ψt, envt, _ = leading_boundary(ψt0, 𝕋n1, optim_alg)

    return normality, f_history, galerkin_history
end

maxiter=200
χ = 16
normality0, f_history0, galerkin_history0 = VUMPS_history(0, χ, maxiter)
normality0_125, f_history0_125, galerkin_history0_125 = VUMPS_history(0.125, χ, maxiter)
normality0_25, f_history0_25, galerkin_history0_25 = VUMPS_history(0.25, χ, maxiter)
normality0_5, f_history0_5, galerkin_history0_5 = VUMPS_history(0.5, χ, maxiter)
normality1, f_history1, galerkin_history1 = VUMPS_history(1, χ, maxiter)
normality2, f_history2, galerkin_history2 = VUMPS_history(2, χ, maxiter)
normality3, f_history3, galerkin_history3 = VUMPS_history(3, χ, maxiter)
@save "gauge_AF_triangular_ising/data/VUMPS_history_chi$(χ)" normality0 f_history0 galerkin_history0 normality0_125 f_history0_125 galerkin_history0_125 normality0_25 f_history0_25 galerkin_history0_25 normality0_5 f_history0_5 galerkin_history0_5 normality1 f_history1 galerkin_history1 normality2 f_history2 galerkin_history2 normality3 f_history3 galerkin_history3

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
line0 = lines!(ax1, 1:length(f_history0), abs.(f_history0 .- f_exact) ./ f_exact, label=L"\tau=0")
line0_125 = lines!(ax1, 1:length(f_history0_125), abs.(f_history0_125 .- f_exact) ./ f_exact, label=L"\tau=0.125")
line0_25 = lines!(ax1, 1:length(f_history0_25), abs.(f_history0_25 .- f_exact) ./ f_exact, label=L"\tau=0.25")
line0_5 = lines!(ax1, 1:length(f_history0_5), abs.(f_history0_5 .- f_exact) ./ f_exact, label=L"\tau=0.5")
line1 = lines!(ax1, 1:length(f_history1), abs.(f_history1 .- f_exact) ./ f_exact, label=L"\tau=1")
line2 = lines!(ax1, 1:length(f_history2), abs.(f_history2 .- f_exact) ./ f_exact, label=L"\tau=2")
line3 = lines!(ax1, 1:length(f_history3), abs.(f_history3 .- f_exact) ./ f_exact, label=L"\tau=3")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{convergence measure}", yscale=log10)
line0 = lines!(ax2, 1:length(f_history0), galerkin_history0, label=L"\tau=0")
line0_125 = lines!(ax2, 1:length(f_history0_125), galerkin_history0_125, label=L"\tau=0.125")
line0_25 = lines!(ax2, 1:length(f_history0_25), galerkin_history0_25, label=L"\tau=0.25")
line0_5 = lines!(ax2, 1:length(f_history0_5), galerkin_history0_5, label=L"\tau=0.5")
line1 = lines!(ax2, 1:length(f_history1), galerkin_history1, label=L"\tau=1")
line2 = lines!(ax2, 1:length(f_history2), galerkin_history2, label=L"\tau=2")
line3 = lines!(ax2, 1:length(f_history3), galerkin_history3, label=L"\tau=3")
axislegend(ax2)
@show fig 

save("gauge_AF_triangular_ising/data/VUMPS_history_chi$(χ).pdf", fig)

function VUMPS_plus_gradoptim_history(τ::Real, χ::Int, maxiter::Int)
    normality, 𝕋n1 = f_normality(τ)
    @show norm(𝕋n1.opp[1])
    ψt0 = InfiniteMPS([ℂ^2], [ℂ^χ])

    f_history = Float64[]
    galerkin_history = Float64[]
    function finalize1(iter,state,H,envs)
        st = convert(InfiniteMPS,state) 
        f1 = log(norm(dot(st, 𝕋n1, st)))
        push!(f_history, f1)
        push!(galerkin_history, MPSKit.calc_galerkin(state, envs))
        @show length(f_history), length(galerkin_history)
        return (state, envs)
    end
    function finalize2(x, f, g, numiter)
        st = convert(InfiniteMPS,x.state) 
        f1 = log(norm(dot(st, 𝕋n1, st)))
        push!(f_history, f1)
        push!(galerkin_history, sqrt(norm(MPSKit.GrassmannMPS.inner(g[1], g[1]))))
        @show length(f_history), length(galerkin_history)
        return (x, f, g)
    end

    optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=maxiter÷2, finalize=finalize1)
    optim_alg2 = GradientGrassmann(;tol=1e-12, maxiter=maxiter÷2, finalize! = finalize2)
    ψt1, envt, _ = leading_boundary(ψt0, 𝕋n1, optim_alg1)
    ψt, envt, _ = leading_boundary(ψt1, 𝕋n1, optim_alg2)

    return normality, f_history, galerkin_history
end

maxiter=400
χ = 64
#normality0, f_history0, galerkin_history0 = VUMPS_plus_gradoptim_history(0, χ, maxiter)
#normality0_125, f_history0_125, galerkin_history0_125 = VUMPS_plus_gradoptim_history(0.125, χ, maxiter)
#normality0_25, f_history0_25, galerkin_history0_25 = VUMPS_plus_gradoptim_history(0.25, χ, maxiter)
#normality0_5, f_history0_5, galerkin_history0_5 = VUMPS_plus_gradoptim_history(0.5, χ, maxiter)
normality1, f_history1, galerkin_history1 = VUMPS_plus_gradoptim_history(1, χ, maxiter)
normality2, f_history2, galerkin_history2 = VUMPS_plus_gradoptim_history(2, χ, maxiter)
normality3, f_history3, galerkin_history3 = VUMPS_plus_gradoptim_history(3, χ, maxiter)
@save "gauge_AF_triangular_ising/data/VUMPS_plus_gradoptim_history_chi$(χ)" normality1 f_history1 galerkin_history1 normality2 f_history2 galerkin_history2 normality3 f_history3 galerkin_history3

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
#line0 = lines!(ax1, 1:length(f_history0), abs.(f_history0 .- f_exact) ./ f_exact, label=L"\tau=0")
#line0_125 = lines!(ax1, 1:length(f_history0_125), abs.(f_history0_125 .- f_exact) ./ f_exact, label=L"\tau=0.125")
#line0_25 = lines!(ax1, 1:length(f_history0_25), abs.(f_history0_25 .- f_exact) ./ f_exact, label=L"\tau=0.25")
#line0_5 = lines!(ax1, 1:length(f_history0_5), abs.(f_history0_5 .- f_exact) ./ f_exact, label=L"\tau=0.5")
line1 = lines!(ax1, 1:length(f_history1), abs.(f_history1 .- f_exact) ./ f_exact, label=L"\tau=1")
line2 = lines!(ax1, 1:length(f_history2), abs.(f_history2 .- f_exact) ./ f_exact, label=L"\tau=2")
line3 = lines!(ax1, 1:length(f_history3), abs.(f_history3 .- f_exact) ./ f_exact, label=L"\tau=3")
axislegend(ax1)
@show fig 

ax2 = Axis(fig[2, 1], xlabel=L"\text{steps}", ylabel=L"\text{convergence measure}", yscale=log10)
#line0 = lines!(ax2, 1:length(f_history0), galerkin_history0, label=L"\tau=0")
#line0_125 = lines!(ax2, 1:length(f_history0_125), galerkin_history0_125, label=L"\tau=0.125")
#line0_25 = lines!(ax2, 1:length(f_history0_25), galerkin_history0_25, label=L"\tau=0.25")
#line0_5 = lines!(ax2, 1:length(f_history0_5), galerkin_history0_5, label=L"\tau=0.5")
line1 = lines!(ax2, 1:length(f_history1), galerkin_history1, label=L"\tau=1")
line2 = lines!(ax2, 1:length(f_history2), galerkin_history2, label=L"\tau=2")
line3 = lines!(ax2, 1:length(f_history3), galerkin_history3, label=L"\tau=3")
axislegend(ax2)
@show fig 

save("gauge_AF_triangular_ising/data/VUMPS_plus_gradoptim_history_chi$(χ).pdf", fig)