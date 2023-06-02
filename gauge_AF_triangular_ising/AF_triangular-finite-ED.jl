using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using CairoMakie
using JLD2

include("../utils.jl");

T = tensor_triangular_AF_ising()

L = 6
boundary_condition = :pbc

Tdag = mpotensor_dag(T)
𝕋 = mpo_gen(L, T, boundary_condition);
𝕋dag = mpo_gen(L, Tdag, boundary_condition);
𝕋mat = convert_to_mat(𝕋);
𝕋dagmat = convert_to_mat(𝕋dag);

# right eigenvector; double degeneracy
Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
Λm = norm(Λt[end])
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
lines!(ax1, Λm .* cos.(angles), Λm .* sin.(angles), color=:grey)
@show fig
save("AF_triangular_ising/data/percolation-finite-spectrum.pdf", fig)

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

# gauge transformation
σz = TensorMap(ComplexF64[1 0 ; 0 -1], ℂ^2, ℂ^2)
ZZ = σz ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ σz ⊗ σz ⊗ Id2 ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ σz ⊗ σz ⊗ Id2 ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ σz ⊗ Id2 + 
    Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz ⊗ σz + 
    σz ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ Id2 ⊗ σz 
τs = -1:0.1:3
normalities = Float64[]

@show (ψb1' * ZZ * ψb1)[1]
@show (ψt1' * ZZ * ψt1)[1]

for τ in τs
    P = exp(-τ * ZZ)
    Pinv = exp(τ * ZZ)

    𝕋mat_1 = P * 𝕋mat * Pinv
    𝕋dagmat_1 = Pinv * 𝕋dagmat * P

    aaa = 𝕋mat_1 * 𝕋dagmat_1 
    bbb = 𝕋dagmat_1 * 𝕋mat_1 

    normality = log(dot(aaa, bbb) * dot(bbb, aaa)) - log(dot(aaa, aaa)) - log(dot(bbb, bbb))
    push!(normalities, -real(normality) / 6)
end
@show normalities

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"\text{normality measure}", yscale=log10)
lines!(ax1, τs, normalities)
@show fig
save("AF_triangular_ising/data/percolation-finite-spectrum.pdf", fig)

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


