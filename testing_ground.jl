using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

A = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2);
X = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2);
B = TensorMap(rand, ComplexF64, ℂ^2, ℂ^2);
Adata = reshape(A.data, (2,2,2,2))
Adata = Adata + conj.(permutedims(Adata,(1, 3, 2, 4)));
Adata = reshape(Adata, (4,4))
A = TensorMap(Adata, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
permute(A, (1,3,2,4)) 

B = B + B'
B = add_util_leg(B)
X = add_util_leg(X)

𝔸 = DenseMPO([A])
𝔹 = DenseMPO([B])
𝕏 = DenseMPO([X])
𝔹 = 𝕏

eigen(B.data)

𝕋 = 𝔸 * 𝔹
𝕋dag = 𝔹 * 𝔸

#T = tensor_percolation(0.5, 0.5);
#Tdag = mpotensor_dag(T)
#𝕋 = mpo_gen(1, T, :inf)
#𝕋dag = mpo_gen(1, Tdag, :inf)

@tensor t1[-1; -2] := 𝕋.opp[1][1 -1 -2 1]
Λ1, P1 = eigen(t1)
Pinv1 = inv(P1)

hs = zeros(4)
Pinv1.data' * Pinv1.data
hs[1] = log(Pinv1.data' * Pinv1.data)[1] |> real
hs[2] = log(Pinv1.data' * Pinv1.data)[1,2] |> real
hs[3] = log(Pinv1.data' * Pinv1.data)[1,2] |> imag
hs[4] = log(Pinv1.data' * Pinv1.data)[4] |> real
@show [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]] - log(Pinv1.data' * Pinv1.data)
hs

function AAprime_straight(𝕋::DenseMPO, 𝕋dag::DenseMPO, hs::Vector{<:Real})
    Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
    H = TensorMap(Hmat, ℂ^2, ℂ^2)
    H = H + H'
    G = exp(H)
    Ginv = exp(-H)

    𝔾 = mpo_gen(1, add_util_leg(G), :inf)
    𝔾inv = mpo_gen(1, add_util_leg(Ginv), :inf)

    ψ1 = convert(InfiniteMPS, 𝔾 * 𝕋 * 𝔾inv * 𝕋dag * 𝔾)
    ψ2 = convert(InfiniteMPS, 𝕋dag * 𝔾 * 𝕋)

    -norm(dot(ψ1, ψ2))
end

function g_AAprime_straight!(ghs::Vector{<:Real}, hs::Vector{<:Real})
    ghs .= grad(central_fdm(5, 1), x -> AAprime_straight(𝕋, 𝕋dag, x), hs)[1]
end

AAprime_straight(𝕋, 𝕋dag, hs)
using FiniteDifferences, Optim

res = optimize(x -> AAprime_straight(𝕋, 𝕋dag, x), g_AAprime_straight!, hs, LBFGS(), Optim.Options(show_trace = true))

@show Optim.minimum(res)
hs = Optim.minimizer(res)
AAprime_straight(𝕋, 𝕋dag, hs)

Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
H = TensorMap(Hmat, ℂ^2, ℂ^2)
H = H + H'
G = exp(H)
Ginv = exp(-H)

Λ, U = eigen(G)
@show sqrt(Λ) 

"""
T = tensor_triangular_AF_ising_alternative()
T = tensor_triangular_AF_ising()

@load "frustrated_obc_L84.jld" ψms
ψ84 = copy(ψms[end])
for ix in 1:84 
    @show ix, maximum(norm.(ψ84.AL[1].data))
end

@load "frustrated_inf_L1.jld" ψms
ψinf = copy(ψms[end])

maximum(norm.(ψinf.AL[1].data))

@load "frustrated_obc_L30.jld" ψms

ϕ = copy(ψms[end])
L = length(ϕ)
σs = sample_n_domain_wall(ϕ, :frstr, :obc)

P = TensorMap(rand, ComplexF64, ℂ^2, ℂ^2)
P = P / norm(P)

@show eig(P)
P = add_util_leg(P)
ℙ = mpo_gen(L, P, :obc)


σs_P = sample_n_domain_wall(ℙ * ϕ, :frstr, :obc)

plot()
xlabel!("number of domain walls")
ylabel!("number of samples (1000 in total)")
histogram!(σs, alpha=0.5, color=:red, bins=1:L, label="original")
histogram!(σs_P, alpha=0.5, color=:blue, bins=1:L, label="after gauge")


ψ = InfiniteMPS([ℂ^2], [ℂ^10])
ϕ = InfiniteMPS([ℂ^2], [ℂ^10])

dot(ϕ, ψ) 

P = TensorMap(rand, ComplexF64, ℂ^2, ℂ^2)
ℙ = DenseMPO([add_util_leg(P)])

ψ1 = ℙ * ψ
ϕ1 = ℙ * ϕ
dot(ψ1, ϕ1)

L = 40
ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^10)
ϕ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^10)

dot(ψ, ϕ) / norm(ψ) / norm(ϕ) |> norm
ℙ = DenseMPO(fill(add_util_leg(P), L))

ψ1 = ℙ * ψ
ϕ1 = ℙ * ϕ
dot(ψ1, ϕ1) / norm(ψ1) / norm(ϕ1) |> norm
"""