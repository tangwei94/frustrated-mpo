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

f_exact = 0.3230659669

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

    return normality, 𝕋n1, 𝕋ndag1 
end

# MPOs 
τ = 0.5
ℙ = mpo_gen(1, genP(τ), :Inf);
ℙinv = mpo_gen(1, genP(-τ), :Inf);

# normal matrix
Tn0 = tensor_triangular_AF_ising_adapted()
𝕋n0 = DenseMPO([Tn0])
ψt0 = InfiniteMPS([ℂ^2], [ℂ^64])
optim_alg = VUMPS(tol_galerkin=1e-12, maxiter=10000);
ψ0, = leading_boundary(ψt0, 𝕋n0, optim_alg);

# P can also destropy ψ0, (X) since the U1 symmetry is simultaneously broken 
ψa = ℙ * ψ0; 
ψb = ℙinv * ψ0;
dot(ψa, ψ0)
scattering0 = importance_scattering_L(ψa, ψb); 
fig, ax, hm = heatmap(norm.(scattering0.data); colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues, figure = (; resolution=(600, 600)) , axis = (; xlabel=L"i", ylabel=L"j"))

# vomps \tilde{\calP}_2 with τ 
extract_result(results) = (
    [result[1] for result in results],
    [result[2] for result in results],
    [result[3] for result in results],
    [result[4] for result in results],
    [result[5] for result in results]
)
for τ in 2.0:-0.25:0.5
    normality, 𝕋n1, 𝕋ndag1 = f_normality(τ)
    @show normality
    ψ1 = InfiniteMPS([ℂ^2], [ℂ^32])
    ψ2 = InfiniteMPS([ℂ^2], [ℂ^32])
    optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=1000) 
    results_32 = map(1:100) do ix
        global ψ1, ψ2
        ψ1, _ = approximate(ψ1, (𝕋n1, ψ1), optim_alg1)
        ψ2, _ = approximate(ψ2, (𝕋ndag1, ψ2), optim_alg1)
        var1 = log(dot(ψ1, 𝕋ndag1*𝕋n1, ψ1) / dot(ψ1, 𝕋n1, ψ1) / dot(ψ1, 𝕋ndag1, ψ1))
        var2 = log(dot(ψ2, 𝕋n1*𝕋ndag1, ψ2) / dot(ψ2, 𝕋n1, ψ2) / dot(ψ2, 𝕋ndag1, ψ2))
        f = log(dot(ψ2, 𝕋n1, ψ1) / dot(ψ2, ψ1))
        return ψ1, ψ2, f, var1, var2
    end;
    ψ1s_32, ψ2s_32, fs_32, vars1_32, vars2_32 = extract_result(results_32);
    @save "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
end

for τ in [0.25, 0.0]
    @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(0.5)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
    N = length(fs_32)
    ψr = ψ1s_32[end]
    ψl = ψ2s_32[end]

    Δτ = 0.5 - τ
    ℙ = mpo_gen(1, genP(Δτ), :Inf);
    ℙinv = mpo_gen(1, genP(-Δτ), :Inf);
    optim_alg1 = VUMPS(tol_galerkin=1e-12, maxiter=1000) 
    ψl, _ = approximate(ψl, (ℙ, ψl), optim_alg1)
    ψr, _ = approximate(ψr, (ℙinv, ψr), optim_alg1)
    @save "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
end

for τ in 2.0:-0.25:0.
    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        N = length(fs_32)

        fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 300))
        ax1 = Axis(fig[1, 1], xlabel=L"\text{steps}", ylabel=L"\text{error in }f", yscale=log10)
        lines!(ax1, 1:N, abs.(fs_32 .- f_exact) ./ f_exact, label=L"χ=32")
        text!(ax1, 0, 10^1, text=L"\text{(a)}", align=(:left, :top))
        @show fig 

        ax2 = Axis(fig[1, 2], xlabel=L"\text{steps}", ylabel=L"\text{variance}", yscale=log10)
        lines!(ax2, 1:N, norm.(vars1_32), label=L"|\psi^{\mathrm{R}}\rangle")
        lines!(ax2, 1:N, norm.(vars2_32), label=L"|\psi^{\mathrm{L}}\rangle")
        text!(ax2, 0, 10^(0), text=L"\text{(b)}", align=(:left, :top))
        axislegend(ax2, position=:rt)
        save("gauge_AF_triangular_ising/data/fig-VOMPS_tau$(τ)_plot.pdf", fig)
        @show fig 
    end

    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        ψr = ψ1s_32[end]
        ψl = ψ2s_32[end]
    else 
        @load "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
    end
    scattering = importance_scattering_L(ψl, ψr); 

    λsvdr = svdvals(ψr.CR[1].data)
    λsvdl = svdvals(ψl.CR[1].data)
    λr = λsvdr .^ 2  / sum(λsvdr .^ 2)
    λl = λsvdl .^ 2  / sum(λsvdl .^ 2)
    ρlr = rhoLR(ψl, ψr)
    λlr = reverse(eigvals(ρlr.data))
    @show τ, λlr

    fig, ax, hm = heatmap(norm.(scattering.data); colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues, figure = (; resolution=(600, 600)) , axis = (; xlabel=L"i", ylabel=L"j"))
    ax = Axis(fig[1, 2], xlabel=L"i", ylabel=L"j") 
    heatmap!(ax, norm.(ρlr.data), colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues)
    Colorbar(fig[:, end+1], hm)
    ax = Axis(fig[2, 1:end], xlabel=L"\text{index}", ylabel=L"\lambda")
    lines!(ax, log10.(norm.(λlr)), linestyle=:dash, label="λlr, n")
    save("gauge_AF_triangular_ising/data/tmpfig-VOMPS_tau$(τ)_plot2.pdf", fig)
    @show fig
end

# plot together 
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
gf = fig[1:4, 1:6] = GridLayout()
gl = fig[end+1, 2:5] = GridLayout()
ax2 = Axis(gf[1, 1], xlabel=L"\text{index}", ylabel=L"\lambda", yscale=log10)
ax3 = Axis(gf[1, 2], xlabel=L"\text{index}", ylabel=L"\lambda", yscale=log10)

EEs, EELs, EERs = Float64[], Float64[], Float64[]
τs = 1.75:-0.25:0.0
for τ in τs
    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        N = length(fs_32)
        ψr = ψ1s_32[end]
        ψl = ψ2s_32[end]
    else
        @load "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
    end

    λsvdr = svdvals(ψr.CR[1].data)
    λsvdl = svdvals(ψl.CR[1].data)
    λr = λsvdr .^ 2  / sum(λsvdr .^ 2)
    λl = λsvdl .^ 2  / sum(λsvdl .^ 2)
    ρlr = rhoLR(ψl, ψr)
    λlr = reverse(eigvals(ρlr.data))
    push!(EEs, real(get_EE(λlr)))
    push!(EELs, real(get_EE(λl)))
    push!(EERs, real(get_EE(λr)))
    scatter!(ax2, norm.(λr), label=latexstring("\$τ=$(τ)\$"))
    lines!(ax2, norm.(λr), linestyle=:dash, label=latexstring("\$τ=$(τ)\$"))
    scatter!(ax3, norm.(λl), label=latexstring("\$τ=$(τ)\$"))
    lines!(ax3, norm.(λl), linestyle=:dash, label=latexstring("\$τ=$(τ)\$"))
end
ylims!(ax2, (1e-10, 10^1))
ylims!(ax3, (1e-10, 10^1))
text!(ax2, 0, 10^0.9, text=L"\text{(a)}", align=(:left, :top))
text!(ax3, 0, 10^0.9, text=L"\text{(b)}", align=(:left, :top))
Legend(gl[1, 1], ax2, nbanks=4, merge=true)
save("gauge_AF_triangular_ising/data/fig-single-side-rho-VOMPS.pdf", fig)
@show fig

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 300))
ax = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"S_{\mathrm{EE}}")
lines!(ax, τs, EELs, linestyle=:dash, label=L"|\psi^{\mathrm{L}}\rangle")
scatter!(ax, τs, EELs, label=L"|\psi^{\mathrm{L}}\rangle")
lines!(ax, τs, EERs, linestyle=:dash, label=L"|\psi^{\mathrm{R}}\rangle")
scatter!(ax, τs, EERs, label=L"|\psi^{\mathrm{R}}\rangle")
axislegend(ax, position=:rb, merge=true)
save("gauge_AF_triangular_ising/data/fig-single-side-SE-VOMPS.pdf", fig)
@show fig

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
gf1 = fig[1, 1:5] = GridLayout()
gl = fig[1, end+1] = GridLayout()
ax2 = Axis(gf1[1, 1], xlabel=L"\text{index}", ylabel=L"\lambda", yscale=log10)
ax3 = Axis(gf1[2, 1], xlabel=L"\text{index}", ylabel=L"\lambda")
for τ in 1.75:-0.25:0.0
    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        N = length(fs_32)
        ψr = ψ1s_32[end]
        ψl = ψ2s_32[end]
    else
        @load "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
    end

    ρlr = rhoLR(ψl, ψr)
    λlr = reverse(eigvals(ρlr.data))
    scatter!(ax2, norm.(real.(λlr)), label=latexstring("\$τ=$(τ)\$"))
    lines!(ax2, norm.(real.(λlr)), linestyle=:dash, label=latexstring("\$τ=$(τ)\$"))
    @show sum(imag.(λlr))
    scatter!(ax3, imag.(λlr), label=latexstring("\$τ=$(τ)\$"))
end
ylims!(ax2, (1e-10, 10^1))
#ylims!(ax3, (1e-10, 10^1))
Legend(gl[1, 1], ax2, nbanks=1, merge=true)
save("gauge_AF_triangular_ising/data/fig-double-side-rho-VOMPS.pdf", fig)
@show fig

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 1200))
for (ix, iy, τ, label) in zip([1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 1, 2, 1, 2, 1, 2], 1.75:-0.25:0, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"])
    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        N = length(fs_32)
        ψr = ψ1s_32[end]
        ψl = ψ2s_32[end]
    else
        @load "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
    end

    ax2 = PolarAxis(fig[ix, iy], rlimits = (1, 10), rticks = ([3, 6, 9], ["10⁻⁶", "10⁻³", "10⁰"]), rtickangle=pi/2, thetaticklabelsvisible=false, title=latexstring("\\text{$(label) } \\tau=$(τ)"), titlealign=:left, titlegap=2, thetaticklabelpad=-10, rticklabelpad=-10, thetagridcolor=:lightgrey, rgridcolor=:lightgrey)
    ρlr = rhoLR(ψl, ψr)
    λlr = reverse(eigvals(ρlr.data))
    @show λlr
    f_step(x) = x * (sign(x) + 1) / 2
    scatter!(ax2, angle.(λlr), f_step.(log10.(norm.(λlr)) .+ 9), color=:skyblue2)
    tightlimits!(ax2) 
end
#Legend(gl[1, 1], ax2, nbanks=1, merge=true)
save("gauge_AF_triangular_ising/data/fig-double-side-rho-VOMPS-phases.pdf", fig)
@show fig

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 300))
ax2 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"S_{\mathrm{EE}}")
lines!(ax2, τs, EEs, linestyle=:dash)
scatter!(ax2, τs, EEs)
save("gauge_AF_triangular_ising/data/fig-double-side-SE-VOMPS.pdf", fig)
@show fig

###########################################
τ = 0.5
    if τ ≥ 0.5
        @load "gauge_AF_triangular_ising/data/vomps_chi32_tau$(τ)_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32
        N = length(fs_32)
        ψr = ψ1s_32[end]
        ψl = ψ2s_32[end]
    else
        @load "gauge_AF_triangular_ising/data/fixedpoint_chi32_tau$(τ)_results.jld2" ψl ψr
    end

λsvdr = svdvals(ψr.CR[1].data)
λsvdl = svdvals(ψl.CR[1].data)
λr = λsvdr .^ 2  / sum(λsvdr .^ 2)
λl = λsvdl .^ 2  / sum(λsvdl .^ 2)
ρlr = rhoLR(ψl, ψr)
λlr = reverse(eigvals(ρlr.data))

sort(real.(diag(ρlr.data)))

λlr[1]
λlr[end]

# the boundary MPS of \calP_2 


# plot. 
fig, ax, hm = heatmap(norm.(scattering.data); colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues, figure = (; resolution=(600, 250)) , axis = (; xlabel=L"i", ylabel=L"j"))
ax = Axis(fig[1, 2], xlabel=L"i", ylabel=L"j") 
heatmap!(ax, norm.(scattering2.data), colorrange=(1e-3, 1), colorscale=log10, colormap=:Blues)
Colorbar(fig[:, end+1], hm)
save("gauge_AF_triangular_ising/data/fig-Mmatrix_T2.pdf", fig)
@show fig

#### VOMPS 
# load VOMPS result.
@load "gauge_AF_triangular_ising/data/vomps_chi32_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32

_, ix1 = findmin(real.(vars1_32))
_, ix2 = findmin(real.(vars2_32))
ψ1, ψ2 = ψ1s_32[ix1], ψ2s_32[ix2];
scattering = importance_scattering_L(ψ2, ψ1);
@load "gauge_AF_triangular_ising/data/vomps_chi32_tau0_5_results.jld2" ψ1s_32 ψ2s_32 fs_32 vars1_32 vars2_32

# the M-matrix differs from expectation
fig, ax, hm = heatmap(log10.(norm.(scattering.data)), colorrange=(-3, 0), colormap=:Blues)
Colorbar(fig[:, end+1], hm)
@show fig

# compare with the result obtained from the gauged MPO
norm(dot(ψ1, ψr1))

ψr1_t = changebonds(ψr1, SvdCut(truncdim(32)))
ψl1_t = changebonds(ψl1, SvdCut(truncdim(32)))

var(ψx) = log(dot(ψx, 𝕋dag*𝕋, ψx) / dot(ψx, 𝕋, ψx) / dot(ψx, 𝕋dag, ψx))
vardag(ψx) = log(dot(ψx, 𝕋*𝕋dag, ψx) / dot(ψx, 𝕋, ψx) / dot(ψx, 𝕋dag, ψx))
var2(ψx) = log(dot(ψx, 𝕋ndag1*𝕋n1, ψx) / dot(ψx, 𝕋n1, ψx) / dot(ψx, 𝕋ndag1, ψx))
vardag2(ψx) = log(dot(ψx, 𝕋n1*𝕋ndag1, ψx) / dot(ψx, 𝕋n1, ψx) / dot(ψx, 𝕋ndag1, ψx))
var(ψr1_t)
vardag(ψl1_t)

#### entanglement spectrum
λsvdr1 = svdvals(ψr1.CR[1].data)
λsvdl1 = svdvals(ψl1.CR[1].data)
λr1 = λsvdr1 .^ 2  / sum(λsvdr1 .^ 2)
λl1 = λsvdl1 .^ 2  / sum(λsvdl1 .^ 2)
ρlr1 = rhoLR(ψl1, ψr1)
λlr1 = reverse(eigvals(ρlr1.data))

λsvdr = svdvals(ψr.CR[1].data)
λsvdl = svdvals(ψl.CR[1].data)
λr = λsvdr .^ 2  / sum(λsvdr .^ 2)
λl = λsvdl .^ 2  / sum(λsvdl .^ 2)
ρlr = rhoLR(ψl, ψr)
λlr = reverse(eigvals(ρlr.data))

λsvdr2 = svdvals(ψr2.CR[1].data)
λsvdl2 = svdvals(ψl2.CR[1].data)
λr2 = λsvdr2 .^ 2  / sum(λsvdr2 .^ 2)
λl2 = λsvdl2 .^ 2  / sum(λsvdl2 .^ 2)
ρlr2 = rhoLR(ψl1, ψr2)
λlr2 = reverse(eigvals(ρlr2.data))

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax = Axis(fig[1, 1], xlabel=L"\text{index}", ylabel=L"\lambda")
lines!(ax, log10.(λl1), label="λl1, n")
lines!(ax, log10.(λr1), label="λr1, n")
lines!(ax, log10.(norm.(λlr1)), label="λlr1, n")

lines!(ax, log10.(λl2), label="λl2, n")
lines!(ax, log10.(λr2), label="λr2, n")
lines!(ax, log10.(norm.(λlr2)), label="λlr2, n")

#lines!(ax, log10.(λl), linestyle=:dash, label="λl, n")
#lines!(ax, log10.(λr), linestyle=:dash, label="λr, n")
#lines!(ax, log10.(norm.(λlr)), linestyle=:dash, label="λlr, n")
axislegend(ax, position=:lb)
@show fig



fig, ax, hm = heatmap(log10.(norm.(ρlr2.data)), colorrange=(-6, 0), colormap=:Blues)
Colorbar(fig[:, end+1], hm)
@show fig