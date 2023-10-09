using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = DenseMPO([T])
𝕋dag = DenseMPO([Tdag])

σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
σp = TensorMap(ComplexF64[0 1; 0 0], ℂ^2, ℂ^2)
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
    #𝕋n1 = DenseMPO([Tn1])
    #𝕋ndag1 = DenseMPO([Tndag1])

    normality = mpo_ovlp1(𝕋n1 * 𝕋ndag1, 𝕋ndag1 * 𝕋n1)

    return normality, 𝕋n1
end

_, 𝕋n_nf = f_normality(Inf)

normalities = Float64[]
fidelities_with_Tnf = Float64[]
τs = -0.5:0.1:2.5
for τ in τs 
    y, 𝕋tmp = f_normality(τ)

    push!(normalities, real(y))
    push!(fidelities_with_Tnf, real(mpo_ovlp1(𝕋tmp, 𝕋n_nf)))
    @show τ, y, log(y)
end

f_exact = 0.3230659669

function VUMPS_history(τ::Real, χ::Int, maxiter::Int)
    normality, 𝕋n1 = f_normality(τ)
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

maxiter=250
χ = 64
normality0, f_history0, galerkin_history0 = VUMPS_history(0, χ, maxiter);
normality1, f_history1, galerkin_history1 = VUMPS_history(1, χ, maxiter);
normality2, f_history2, galerkin_history2 = VUMPS_history(2, χ, maxiter);
@save "gauge_AF_triangular_ising/data/VUMPS_history_chi$(χ)_tau$(τ).jld2" normality=normality0 f_history=f_history0 galerkin_history=galerkin_history0
@save "gauge_AF_triangular_ising/data/VUMPS_history_chi$(χ)_tau$(τ).jld2" normality=normality1 f_history=f_history1 galerkin_history=galerkin_history1
@save "gauge_AF_triangular_ising/data/VUMPS_history_chi$(χ)_tau$(τ).jld2" normality=normality2 f_history=f_history2 galerkin_history=galerkin_history2

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax2 = Axis(fig[1, 1], ylabel=L"1-F(\tilde{\mathcal{T}}_2, \mathcal{T}_1)", yscale=log10, xticklabelsvisible=false) 
lines!(ax2, τs, 1 .- fidelities_with_Tnf)
lines!(ax2, τs, exp.(-8 .* τs), linestyle=:dash)
text!(ax2, -0.5, 10^1.75, text=L"\text{(a)}", align=(:left, :top))
text!(ax2, 2, 10^(-7.), text=L"\mathrm{e}^{-8\tau}", align=(:left, :bottom), color=:darkorange2)
ax1 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"1-F(\tilde{\mathcal{T}}_2 \tilde{\mathcal{T}}_2^\dagger, \tilde{\mathcal{T}}_2^\dagger \tilde{\mathcal{T}}_2)", yscale=log10) 
lines!(ax1, τs, 1 .- normalities)
lines!(ax1, τs, exp.(-8 .* τs), linestyle=:dash)
text!(ax1, -0.5, 10^1.75, text=L"\text{(b)}", align=(:left, :top))
text!(ax1, 2, 10^(-7.), text=L"\mathrm{e}^{-8\tau}", align=(:left, :bottom), color=:darkorange2)
ax3 = Axis(fig[1:2, 2], xlabel=L"\text{steps}", ylabel=L"\text{convergence measure}", yscale=log10)
lines!(ax3, galerkin_history0, label=L"\tau=0")
lines!(ax3, galerkin_history1, label=L"\tau=1")
lines!(ax3, galerkin_history2, label=L"\tau=2")
text!(ax3, -0, 10^1.5, text=L"\text{(c)}", align=(:left, :top))
axislegend(ax3, position=:rt)
save("gauge_AF_triangular_ising/data/fig-gauged-frustrated-mpo-normal-meas.pdf", fig)
@show fig

