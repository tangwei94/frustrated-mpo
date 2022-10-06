using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

χs = [2, 4, 8]

fr, varsr, diffsr, ψmsr = power_projection(𝕋, χs; Npower=30, filename="tempr");
fl, varsl, diffsl, ψmsl = power_projection(𝕋dag, χs; Npower=30, filename="templ");

ψR = ψmsr[end]
ψL = ψmsl[end]

plot()
plot!(varsr, yaxis=:log)
plot!(varsl, yaxis=:log)

function S_straight(ψR::InfiniteMPS, ψL::InfiniteMPS, hs::Vector{<:Real})
    Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
    H = TensorMap(Hmat, ℂ^2, ℂ^2)
    H = H + H'
    G = exp(H)
    Ginv = exp(-H)

    𝔾 = mpo_gen(1, add_util_leg(G), :inf)
    𝔾inv = mpo_gen(1, add_util_leg(Ginv), :inf)

    return log(abs(dot(ψL, 𝔾inv, ψL) * dot(ψR, 𝔾, ψR)) / abs(dot(ψL, ψR))^2)
end

function g_S_straight!(ghs::Vector{<:Real}, hs::Vector{<:Real})
    ghs .= grad(central_fdm(5, 1), x -> S_straight(ψR, ψL, x), hs)[1]
end

hs = zeros(4)

S_straight(ψR, ψL, hs)

using FiniteDifferences, Optim

res = optimize(x -> S_straight(ψR, ψL, x), g_S_straight!, hs, LBFGS(), Optim.Options(show_trace = true))

hs = Optim.minimizer(res)

Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
H = TensorMap(Hmat, ℂ^2, ℂ^2)
H = H + H'
G = exp(H)
Ginv = exp(-H)

Λ, U = eigen(G)

P = sqrt(Λ) * U'
Pinv = inv(P)
Pdag = P'
Pinvdag = Pinv'

ℙ = DenseMPO(add_util_leg(P))
ℙinv = DenseMPO(add_util_leg(Pinv))
ℙdag = DenseMPO(add_util_leg(Pdag))
ℙinvdag = DenseMPO(add_util_leg(Pinvdag))

𝕋1 = ℙ * 𝕋 * ℙinv
𝕋1dag = ℙinvdag * 𝕋dag * ℙdag

fr, varsr, diffsr, ψmsr1 = power_projection(𝕋1, χs; Npower=30, filename="tempr");
fl, varsl, diffsl, ψmsl1 = power_projection(𝕋1dag, χs; Npower=30, filename="templ");

ψR1 = ψmsr1[end]
ψL1 = ψmsl1[end]

plot()
plot!(varsr, yaxis=:log)
plot!(varsl, yaxis=:log)

function g_S_straight!(ghs::Vector{<:Real}, hs::Vector{<:Real})
    ghs .= grad(central_fdm(5, 1), x -> S_straight(ψR1, ψL1, x), hs)[1]
end

hs = rand(4)

S_straight(ψR, ψL, hs)
S_straight(ψR1, ψL1, hs)

using FiniteDifferences, Optim

res1 = optimize(x -> S_straight(ψR1, ψL1, x), g_S_straight!, hs, LBFGS(), Optim.Options(show_trace = true))

hs = Optim.minimizer(res1)

Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
H = TensorMap(Hmat, ℂ^2, ℂ^2)
H = H + H'
G = exp(H)
Ginv = exp(-H)

Λ, U = eigen(G)

P = sqrt(Λ) * U'
Pinv = inv(P)
Pdag = P'
Pinvdag = Pinv'

ℙ = DenseMPO(add_util_leg(P))
ℙinv = DenseMPO(add_util_leg(Pinv))
ℙdag = DenseMPO(add_util_leg(Pdag))
ℙinvdag = DenseMPO(add_util_leg(Pinvdag))

𝕋2 = ℙ * 𝕋1 * ℙinv
𝕋2dag = ℙinvdag * 𝕋1dag * ℙdag
χs1 = [2, 4, 8, 16, 32]
fr, varsr, diffsr, ψmsr3 = power_projection(𝕋2, χs1; Npower=30, filename="tempr");
fl, varsl, diffsl, ψmsl3 = power_projection(𝕋2dag, χs1; Npower=30, filename="templ");

ψR2 = ψmsr2[end]
ψL2 = ψmsl2[end]

plot()
plot!(varsr, yaxis=:log)
plot!(varsl, yaxis=:log)

function g_S_straight!(ghs::Vector{<:Real}, hs::Vector{<:Real})
    ghs .= grad(central_fdm(5, 1), x -> S_straight(ψR2, ψL2, x), hs)[1]
end

hs = zeros(4)

S_straight(ψR, ψL, hs)
S_straight(ψR1, ψL1, hs)
S_straight(ψR2, ψL2, hs)

using FiniteDifferences, Optim

res1 = optimize(x -> S_straight(ψR1, ψL1, x), g_S_straight!, hs, LBFGS(), Optim.Options(show_trace = true))

hs = Optim.minimizer(res1)