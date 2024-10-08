using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit, FiniteDifferences
using CairoMakie
using JLD2

include("../utils.jl");

# bond DP
p = 0.6447
p1, p2 = p, p*(2-p)
T = tensor_percolation(p1, p2)

L = 6
boundary_condition = :pbc

Tdag = mpotensor_dag(T)
𝕋 = mpo_gen(L, T, boundary_condition);
𝕋dag = mpo_gen(L, Tdag, boundary_condition);
𝕋mat = convert_to_mat(𝕋);
𝕋dagmat = convert_to_mat(𝕋dag);

# right eigenvector; double degeneracy for CDP
Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
δt1 = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt1.data[end] = 1
ψt1 = Ut * δt1

### check entanglement -> product state
_, S1, _ = tsvd(ψt1, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
@show S1

### finite spectrum
angles = 0:0.01*π:2*π

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\mathrm{Re} λ_n", ylabel=L"\mathrm{Im} λ_n")
scatter1 = scatter!(ax1, real.(Λt), imag.(Λt), marker=:circle, markersize=7.5)
lines!(ax1, cos.(angles), sin.(angles), color=:grey)
@show fig
save("percolation/data/percolation-finite-spectrum.pdf", fig)

# left eigenvector
Λb, Ub = eig(𝕋dagmat);
Λb = diag(Λb.data)
δb1 = Tensor(zeros, ComplexF64, ℂ^length(Λb));
δb1.data[end] = 1;
ψb1 = Ub * δb1

### check entanglement 
_, S1, _ = tsvd(ψb1, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
@show S1

### sigma X 
Id2 = id(ℂ^2)
σx = TensorMap(ComplexF64[0 1 ; 1 0], ℂ^2, ℂ^2)
X = σx ⊗ σx ⊗ σx ⊗ σx ⊗ σx ⊗ σx 
@show X * 𝕋dagmat * X - 𝕋dagmat |> norm
@show (ψb1' * ψb1)[1]
@show (ψb1' * X * ψb1)[1] / (ψb1' * ψb1)[1]

Xeff = [(ψb1' * X * ψb1)[1] (ψb1' * X * ψb2)[1]; (ψb2' * X * ψb1)[1] (ψb2' * X * ψb2)[1]]
Neff = [(ψb1' * ψb1)[1] (ψb1' * ψb2)[1]; (ψb2' * ψb1)[1] (ψb2' * ψb2)[1]]
eigen(Hermitian(sqrt(inv(Neff)) * Xeff * sqrt((inv(Neff)))))

# gauge transformation
σz = TensorMap(ComplexF64[1 0 ; 0 -1], ℂ^2, ℂ^2)
σx = TensorMap(ComplexF64[0 1 ; 1 0], ℂ^2, ℂ^2)
σy = TensorMap(ComplexF64[0 im ; -im 0], ℂ^2, ℂ^2)
X = σx ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σx ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σx ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σx ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σx ⊗ Id2 +
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σx 
Y = σy ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σy ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σy ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σy ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σy ⊗ Id2 +
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σy 
Z = σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ Id2 +
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz 
τs = -3:0.01:3
normalities1 = Float64[]
for τ in τs
    P = exp(-τ * X)
    Pinv = exp(τ * X)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    push!(normalities1, -real(normality) / 6)
end
@show normalities1
τ1 = τs[findmin(normalities1)[2]]

normalities2 = Float64[]
for τ in τs
    P = exp(-τ1 * X - τ * Z)
    Pinv = exp(τ1 * X + τ * Z)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    push!(normalities2, -real(normality) / 6)
end
@show findmin(normalities1), findmin(normalities2)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{normality measure}", yscale=log10)
lines!(ax1, τs, normalities1)
@show fig
ax2 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"\text{normality measure}", yscale=log10)
lines!(ax2, τs, normalities2)
@show fig

function f_normality(coeffs::Vector{<:Real})
    x, z = coeffs

    P = exp(-x * X - z * Z)
    Pinv = exp(x * X + z * Z)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    return -real(normality) / 6
end

function _fg(coeffs::Vector{<:Real})
    fvalue = f_normality(coeffs)
    grad_coeffs = grad(central_fdm(5, 1), f_normality, coeffs)[1]
    return fvalue, grad_coeffs
end

optimize(_fg, Float64[0,0], LBFGS(; maxiter=100, gradtol=1e-12, verbosity=2))