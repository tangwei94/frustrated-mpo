using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using CairoMakie
using JLD2

include("../utils.jl");

p1, p2 = 0.25, 1
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
δt2 = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt2.data[end-1] = 1
ψt1 = Ut * δt1
ψt2 = Ut * δt2

### check entanglement -> product state
_, S1, _ = tsvd(ψt1, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
_, S2, _ = tsvd(ψt2, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
@show S1, S2

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
δb2 = Tensor(zeros, ComplexF64, ℂ^length(Λb));
δb2.data[end-1] = 1;
ψb1 = Ub * δb1
ψb2 = Ub * δb2

### check entanglement 
_, S1, _ = tsvd(ψb1, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
_, S2, _ = tsvd(ψb2, (1, 2, 3), (4, 5, 6); trunc=truncerr(1e-9));
@show S1, S2

### sigma X 
Id2 = id(ℂ^2)
σx = TensorMap(ComplexF64[0 1 ; 1 0], ℂ^2, ℂ^2)
X = σx ⊗ σx ⊗ σx ⊗ σx ⊗ σx ⊗ σx 
@show X * 𝕋dagmat * X - 𝕋dagmat |> norm
@show (ψb1' * ψb1)[1]
@show (ψb2' * ψb2)[1]
@show (ψb1' * ψb2)[1]
@show (ψb1' * X * ψb1)[1] / (ψb1' * ψb1)[1]
@show (ψb2' * X * ψb2)[1] / (ψb2' * ψb2)[1]

Xeff = [(ψb1' * X * ψb1)[1] (ψb1' * X * ψb2)[1]; (ψb2' * X * ψb1)[1] (ψb2' * X * ψb2)[1]]
Neff = [(ψb1' * ψb1)[1] (ψb1' * ψb2)[1]; (ψb2' * ψb1)[1] (ψb2' * ψb2)[1]]
eigen(Hermitian(sqrt(inv(Neff)) * Xeff * sqrt((inv(Neff)))))

# gauge transformation
σz = TensorMap(ComplexF64[1 0 ; 0 -1], ℂ^2, ℂ^2)
ZZ = σz ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σz ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σz ⊗ σz ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ σz ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ σz +
    σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz 
Z = σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ Id2 +
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz 
τs = -1:0.01:3
normalities1 = Float64[]
for τ in τs
    P = exp(-τ * ZZ)
    Pinv = exp(τ * ZZ)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    push!(normalities1, -real(normality) / 6)
end
@show normalities1

normalities2 = Float64[]
for τ in τs
    P = exp(-3 * ZZ - τ * Z)
    Pinv = exp(3 * ZZ + τ * Z)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    push!(normalities2, -real(normality) / 6)
end
@show normalities2

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 600))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{normality measure}", yscale=log10)
lines!(ax1, τs, normalities1)
@show fig
ax2 = Axis(fig[2, 1], xlabel=L"\tau", ylabel=L"\text{normality measure}", yscale=log10)
lines!(ax2, τs, normalities2)
@show fig
save("percolation/data/percolation-finite-spectrum.pdf", fig)
#Id = id(codomain(𝕋mat));

## find Jordan blocks
#for λ in Λt
#    rst_t = findall(x->(abs(x-λ) < 1e-6), Λt)
#    if length(rst_t) != 1
#        S1 = tsvd(𝕋mat - λ * Id)[2].data |> diag
#        S2 = tsvd((𝕋mat - λ * Id) * (𝕋mat - λ * Id))[2].data |> diag
#        @show λ, length(rst_t)
#        @show sum(S1 .> 1e-12), sum(S2 .> 1e-12)
#        #@show S1
#    end
#end


