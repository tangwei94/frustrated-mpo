using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

β = log(1+sqrt(2)) / 2 / 1.2

T, P, Pinv, Pdag, Pdaginv = tensor_trivial(β, 1e-1);

# ED
L = 3
𝕋 = mpo_gen(L, T, :pbc)
ℙ = mpo_gen(L, P, :pbc)
ℙinv = mpo_gen(L, Pinv, :pbc)
ℙdag = mpo_gen(L, Pdag, :pbc)
ℙdaginv = mpo_gen(L, Pdaginv, :pbc)

@show (ℙ * ℙinv).opp[1].data

𝕋mat = convert_to_mat(𝕋);
𝔹mat = convert_to_mat(ℙ*𝕋*ℙinv);
𝔹dagmat = convert_to_mat(ℙdaginv*𝕋*ℙdag);
@show norm(𝔹mat.data' - 𝔹dagmat.data)

Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
δt = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt.data[end] = 1
ψt = Ut * δt;
δ1t = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δ1t.data[end-1] = 1;
ψ1t = Ut * δ1t;
@show dot(ψ1t, ψt);

Λb, Ub = eig(𝔹mat);
Λb = diag(Λb.data)
δb = Tensor(zeros, ComplexF64, ℂ^length(Λb));
δb.data[end] = 1
ψb = Ub * δb;
δ1b = Tensor(zeros, ComplexF64, ℂ^length(Λb));
δ1b.data[end-1] = 1;
ψ1b = Ub * δ1b;
@show dot(ψ1b, ψb);

@show ψbl = δb' * inv(Ub);
@show ψbl * ψb / norm(ψbl) / norm(ψb);

# finite MPS computation
Ls = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
χs = [2, 4, 8]
ψ0s, ψls, ψrs = FiniteMPS[], FiniteMPS[], FiniteMPS[]
fs, fbs = Float64[], Float64[]
for L in Ls
    𝕋 = mpo_gen(L, T, :obc);
    ℙ = mpo_gen(L, P, :obc);
    ℙinv = mpo_gen(L, Pinv, :obc);
    ℙdag = mpo_gen(L, Pdag, :obc);
    ℙdaginv = mpo_gen(L, Pdaginv, :obc);

    𝔹 = ℙ*𝕋*ℙinv;
    𝔹dag = ℙdaginv*𝕋*ℙdag;

    f, vars, diffs, ψms = power_projection(𝕋, χs; Npower=30);
    f1, vars1, diffs1, ψms1 = power_projection(𝔹, χs; Npower=30);
    f2, vars2, diffs2, ψms2 = power_projection(𝔹dag, χs; Npower=30);

    @show dot(ψms1[end], ψms2[end])

    push!(ψ0s, ψms[end])
    push!(ψrs, ψms1[end])
    push!(ψls, ψms2[end])

    push!(fs, real(log(dot(ψms[end], 𝕋, ψms[end]))) / L)
    push!(fbs, real(log(dot(ψms2[end], 𝔹, ψms1[end]) / dot(ψms2[end], ψms1[end]))) / L)
end

# first excited state
ψes, ψles, ψres = FiniteMPS[], FiniteMPS[], FiniteMPS[]
f1s, fb1s = Float64[], Float64[]
χs = [2, 4, 8, 12]
for (L, ψ, ψL, ψR) in zip(Ls, ψ0s, ψls, ψrs)

    𝕋 = mpo_gen(L, T, :obc);
    ℙ = mpo_gen(L, P, :obc);
    ℙinv = mpo_gen(L, Pinv, :obc);
    ℙdag = mpo_gen(L, Pdag, :obc);
    ℙdaginv = mpo_gen(L, Pdaginv, :obc);

    𝔹 = ℙ*𝕋*ℙinv;
    𝔹dag = ℙdaginv*𝕋*ℙdag;

    obtain_1st_excitation_0 = operation_scheme(0.0, 0, [ψ], [ψ]);
    obtain_1st_excitation_R = operation_scheme(0.0, 0, [ψL], [ψL]);
    obtain_1st_excitation_L = operation_scheme(0.0, 0, [ψR], [ψR]);

    _, _, _, ψms = power_projection(𝕋, χs; Npower = 30, operation = obtain_1st_excitation_0);
    _, _, _, ψms1 = power_projection(𝔹, χs; Npower = 30, operation = obtain_1st_excitation_R);
    _, _, _, ψms2 = power_projection(𝔹dag, χs; Npower = 30, operation = obtain_1st_excitation_L);

    push!(ψes, ψms[end])
    push!(ψres, ψms1[end])
    push!(ψles, ψms2[end])

    @show dot(ψms[end], ψ) |> norm
    @show dot(ψms1[end], ψR) |> norm
    @show dot(ψms2[end], ψL) |> norm

    push!(f1s, real(log(dot(ψms[end], 𝕋, ψms[end]))) / L)
    push!(fb1s, real(log(dot(ψms2[end], 𝔹, ψms1[end]) / dot(ψms2[end], ψms1[end]))) / L)
end

plot()
scatter!(Ls[3:end], fs[3:end], alpha=5, markershape=:+, markersize=8, label="f0 hermitian")
scatter!(Ls[3:end], fbs[3:end], alpha=5, markershape=:x, markersize=8, label="f0 non hermitian")
scatter!(Ls[3:end], f1s[3:end], alpha=5, markershape=:+, markersize=8, label="f1 hermitian")
scatter!(Ls[3:end], fb1s[3:end], alpha=5, markershape=:x, markersize=8, label="f1 non hermitian")

ovlps_01_correct = Float64[]
for (L, ψ1, ψ) in zip(Ls, ψ0s, ψes)

    𝕋 = mpo_gen(L, T, :obc);
    ℙ = mpo_gen(L, P, :obc);
    ℙinv = mpo_gen(L, Pinv, :obc);
    ℙdag = mpo_gen(L, Pdag, :obc);
    ℙdaginv = mpo_gen(L, Pdaginv, :obc);

    fidel(x, y) = norm(dot(x, y)) / norm(x) / norm(y)

    push!(ovlps_01_correct, fidel(ℙ * ψ, ℙ * ψ1) )
end

ovlps_01 = [dot(ψr, ψre) for (ψr, ψre) in zip(ψrs, ψres)]

plot()
scatter!(Ls, 1 .- norm.(ovlps_01), yaxis=:log)
scatter!(Ls, 1 .- norm.(ovlps_01_correct), yaxis=:log)

# inifinite MPS computation
L = 1
χs = [2, 4]
𝕋 = DenseMPO(T)
ℙ = DenseMPO(P)
ℙinv = DenseMPO(Pinv)
ℙdag = DenseMPO(Pdag)
ℙdaginv = DenseMPO(Pdaginv)
f, vars, diffs, ψms = power_projection(𝕋, χs; Npower=30);
plot(vars, yaxis=:log)

[entropy(ψms[ix]) for ix in 1:length(χs)]
[entropy(ℙ*ψms[ix]) for ix in 1:length(χs)]

𝔹 = ℙ*𝕋*ℙinv
f1, vars1, diffs1, ψms1 = power_projection(𝔹, χs; Npower=30);
f2, vars2, diffs2, ψms2 = power_projection(ℙdaginv*𝕋*ℙdag, χs; Npower=30);
plot!(vars1, yaxis=:log)
plot!(vars2, yaxis=:log)

[dot(ψms[ix], 𝕋, ψms[ix]) for ix in 1:length(χs)]
[dot(ψms2[ix], 𝔹, ψms1[ix]) / dot(ψms2[ix], ψms1[ix]) for ix in 1:length(χs)]

nonherm_variance!(𝕋 * ψms[end], ψms[end])
nonherm_variance!(𝔹 * ψms1[end], ψms1[end])
nonherm_variance!(𝔹 * ℙ * ψms[end], ℙ * ψms[end])
nonherm_variance!(𝕋 * ℙinv * ψms1[end], ℙinv * ψms1[end])

dot(ψms2[end], ψms1[end])

entropy(ψms[end])
entropy(ψms1[end])

plot()
plot!(vars, yaxis=:log)
plot!(vars1 .+ 1e-16, yaxis=:log)

plot()
plot!(real.(f)[100:end])
plot!(real.(f1)[100:end])

ϕ = ψms1[end]
eltype(ϕ.AL[1])
similar(ϕ.AL[1], MPSKit._firstspace(ϕ.AL[1])←MPSKit._firstspace(ϕ.AL[1]))
similar(ϕ.AL[1])


