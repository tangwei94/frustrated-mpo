using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2

include("../utils.jl");
T = tensor_triangular_AF_ising()
Tdag = tensor_triangular_AF_ising_T()
𝕋 = mpo_gen(1, T, :inf)
𝕋dag = mpo_gen(1, Tdag, :inf)

aaa = convert(InfiniteMPS, 𝕋*𝕋dag)
bbb = convert(InfiniteMPS, 𝕋dag*𝕋)

-norm(dot(aaa, bbb))

@tensor t1[-1; -2] := 𝕋.opp[1][1 -1 -2 1]

#β = log(1+sqrt(2)) / 2 

#T, P, Pinv, Pdag, Pdaginv = tensor_trivial(β, 1e-1);
#P = add_util_leg(TensorMap(rand(2, 2), ℂ^2, ℂ^2))
#Pdag = add_util_leg(TensorMap(Matrix(P.data'), ℂ^2, ℂ^2))
#Pinv = add_util_leg(TensorMap(Matrix(inv(P.data)), ℂ^2, ℂ^2))
#Pdaginv = add_util_leg(TensorMap(Matrix(inv(P.data')), ℂ^2, ℂ^2))
#
#𝔸 = DenseMPO(T)
#ℙ = DenseMPO(P)
#ℙinv = DenseMPO(Pinv)
#ℙdag = DenseMPO(Pdag)
#ℙdaginv = DenseMPO(Pdaginv)
#
#𝕋 = ℙ*𝔸*ℙinv;
#𝕋dag = ℙdaginv*𝔸*ℙdag;

@tensor t1[-1; -2] := 𝕋.opp[1][1 -1 -2 1]
#@tensor p1[-1; -2] := ℙ.opp[1][1 -1 -2 1]
#@tensor pinv1[-1; -2] := ℙinv.opp[1][1 -1 -2 1]

#pinv1 * t1 * p1

Λ1, P1 = eigen(t1)
Pinv1 = inv(P1)
inv(P1) * t1 * P1

#pinv1' * pinv1
# P 可以随意乘unitary，并随意rescale? (不可以随便 rescale) 

hs = zeros(4)
Pinv1.data' * Pinv1.data
hs[1] = log(Pinv1.data' * Pinv1.data)[1] |> real
hs[2] = log(Pinv1.data' * Pinv1.data)[2] |> real
hs[4] = log(Pinv1.data' * Pinv1.data)[4] |> real

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
P = sqrt(Λ) * U'
Pinv = inv(P)
Pdag = U * sqrt(Λ)
Pdaginv = inv(Pdag)

ℙ = DenseMPO([add_util_leg(P)])
ℙinv = DenseMPO([add_util_leg(Pinv)])
ℙdag = DenseMPO([add_util_leg(Pdag)])
ℙdaginv = DenseMPO([add_util_leg(Pdaginv)])
ℕ = ℙ * 𝕋 * ℙinv 
ℕdag = ℙdaginv * 𝕋dag * ℙdag

ψ1 = convert(InfiniteMPS, 𝕋 * 𝕋dag)
ψ2 = convert(InfiniteMPS, 𝕋dag * 𝕋)
dot(ψ1, ψ2)

ψ1 = convert(InfiniteMPS, ℕ * ℕdag)
ψ2 = convert(InfiniteMPS, ℕdag * ℕ)
dot(ψ1, ψ2)

χs = [2, 4, 8, 16]
fs, vars, diffs, ψms = power_projection(ℕ, χs; Npower=30, operation=no_operation)

struct AAprimeStack 
    A::MPOTensor
    H::Matrix # Hermtian  
    isflipped::Bool
end

function MPSKit.transfer_right(vr::AbstractTensorMap, T::AAprimeStack)
    G = exp(H)
    Ginv = exp(-H)
    @tensor vr[-1 -2; -3 -4] := G[1 2] * A[-1 2 4 3] * Ginv[4 5] * A'[5 6 -3 7] * G[7 8] * A'[8 9 -4 10] * G[10 11] * A[-2 11 1 12] * vr[3 12 6 9]
    return vr  
end

function MPSKit.transfer_left(vl::AbstractTensorMap, T::AAprimeStack)
    G = exp(H)
    Ginv = exp(-H)
    @tensor vl[-1 -2; -3 -4] := G[1 2] * A[3 2 4 -3] * Ginv[4 5] * A'[5 -1 6 7] * G[7 8] * A'[8 -2 9 10] * G[10 11] * A[12 11 1 -4]
    return vl
end

Base.:*(T::AAprimeStack, v::AbstractTensorMap) = T(v)

function (T::AAprimeStack)(v::AbstractTensorMap)
    T.isflipped ? MPSKit.transfer_left(v, T) : MPSKit.transfer_right(v, T)
end

function AAprimeStackFidelity(T::AAprimeStack) 
    space_vir = _firstspace(T.A)

    init = similar(T.A, space_vir*space_vir←space_vir*space_vir)
    randomize!(init);

    (vals,vecs,convhist) = eigsolve(TransferMatrix(b.AL,a.AL),init,1,:LM,Arnoldi(krylovdim=krylovdim))
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end

