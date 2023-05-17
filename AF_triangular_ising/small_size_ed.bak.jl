using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("../utils.jl");

L = 6
boundary_condition = :pbc

𝕋 = mpo_gen(L, :frstr, boundary_condition);
𝕋mat = convert_to_mat(𝕋);

Λt, Ut = eig(𝕋mat);
Λt = diag(Λt.data)
δt = Tensor(zeros, ComplexF64, ℂ^length(Λt));
δt.data[end] = 1
ψt = Ut * δt;

# power for the first excited state
#ψtl = δt' * inv(Ut)
#normalize!(ψtl)
#normalize!(ψt)
#ϕt = TensorMap(rand, ComplexF64, codomain(ψt), domain(ψt))
#for ix in 1:500
#    ϕt = normalize(𝕋mat * ϕt) + 0.5*ϕt 
#    ϕt = ϕt - (ψtl * ϕt)[1] * ψtl' 
#    normalize!(ϕt)
#end
#
#norm(𝕋mat * ϕt - Λt[end-1] * ϕt)
#
#norm(𝕋mat * ϕt) * norm(ϕt) / norm(dot(𝕋mat * ϕt, ϕt))
#dot(ϕt, ψt) / norm(ϕt) / norm(ψt) |> norm

𝔾 = mpo_gen(L, :nonfrstr_adapted, boundary_condition);
𝔾mat = convert_to_mat(𝔾);

Λg, Ug = eig(𝔾mat);
Λg = diag(Λg.data)
δg = Tensor(zeros, ComplexF64, ℂ^length(Λg));
δg.data[end] = 1
ψg = Ug * δg;

@show norm(𝔾mat * 𝔾mat' - 𝔾mat' * 𝔾mat) # 𝔾 is normal 
@show norm(𝕋mat * 𝕋mat' - 𝕋mat' * 𝕋mat) # 𝕋 is normal 

@show Λt[end], Λg[end], Λt[end] / Λg[end];
@assert Λt[end] ≈ Λg[end]
Λm = Λt[end]
α = Λt[end] / Λg[end];

length(Λt), length(Λg)
Id = id(codomain(𝕋mat))

# find Jordan blocks
for λ in Λt
    rst_g = findall(x->(abs(x-λ) < 1e-6), Λg)
    rst_t = findall(x->(abs(x-λ) < 1e-6), Λt)
    if length(rst_t) != length(rst_g) 
        @show λ, length(rst_t), length(rst_g)
    end
    if length(rst_t) == length(rst_g) != 1
        S1 = tsvd(𝕋mat - λ * Id)[2].data |> diag
        S2 = tsvd((𝕋mat - λ * Id) * (𝕋mat - λ * Id))[2].data |> diag
        if sum(S1 .> 1e-6) != sum(S2 .> 1e-6)
            @show λ, length(rst_t)
            @show S1, S2
        end
    end
end

segs = (0:100) ./ 100
plot(cos.(2*pi .* segs), sin.(2*pi .* segs), color=:grey, alpha=0.25, linewidth=2, aspect_ratio=:equal, label="")
plot!(cos(2*pi/3) .* segs, sin(2*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, label="")
plot!(cos(4*pi/3) .* segs, sin(4*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, label="")
plot!(cos(0*pi/3) .* segs, sin(0*pi/3) .* segs, color = :grey, alpha=0.25, linewidth=2, label="")

scatter!(real.(Λt ./ Λm), imag.(Λt ./ Λm), markershape=:x, markersize=8, color=:deepskyblue, alpha=0.8, label="𝕋₁ spectrum")
scatter!(real.(Λg ./ Λm), imag.(Λg ./ Λm), markershape=:+, markersize=8, color=:red, alpha=0.8, label="𝕋₂ spectrum")

xlabel!("Re(λ) / λₘ")
ylabel!("Im(λ) / λₘ")

plot!(size=(400, 400))

savefig("fig-"*String(boundary_condition) * "_ED_L$(L).pdf")