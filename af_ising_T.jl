Tf_T = mpo_triangular_AF_ising_T()

Ls = [12, 24, 36];
χs = [4, 8, 16, 24];
for L in Ls
    mpo_frstr = periodic_boundary_conditions(DenseMPO(Tf_T), L);
    fs = ComplexF64[]
    vars = Float64[]
    ψs = []

    ψ = FiniteMPS(L, ℂ^2, ℂ^χs[1]);
    Tψ = mpo_frstr*ψ; 
    for χ in χs
        varm = 1
        ψm = copy(ψ)
        for ix in 1:500
            ψ = changebonds(normalize(Tψ) + 20*ψ, SvdCut(trscheme=truncdim(χ)));
            #ψ = changebonds(Tψ, SvdCut(trscheme=truncdim(χ)))
            normalize!(ψ)
            Tψ = mpo_frstr*ψ 
            f = log(dot(ψ, Tψ)) / L
            var = log(norm(Tψ)^2 / dot(ψ, Tψ) / dot(Tψ, ψ)) / L |> real
            if abs(var) < varm 
                varm = abs(var) 
                ψm = copy(ψ)
            end
            push!(fs, f)
            push!(vars, var)
        end
        push!(ψs, ψm)
        @show L, χ
    end
    @save "tmpdata_T_L$(L).jld" {compress=true} fs vars ψs 
end

@load "tmpdata_T_L36.jld" ψs 
@load "tmpdata_T_L36.jld" vars 
@load "tmpdata_T_L36.jld" fs 
ψs_T_L36 = ψs 
fs_T_L36 = fs 

plot(vars, yaxis=:log)
fs_T_L36[end]
plot(real.(fs))
mpo_frstr = periodic_boundary_conditions(DenseMPO(Tf_T), 36)
ϕ = ψs_T_L36[end]
Tϕ = mpo_frstr * ϕ
log(norm(Tϕ)^2 / dot(ϕ, Tϕ) / dot(Tϕ, ϕ)) / 36 |> real

ϕ1 = ψs_L[end]
Tϕ1 = mpo_frstr * ϕ1
log(norm(Tϕ1)^2 / dot(ϕ1, Tϕ1) / dot(Tϕ1, ϕ1)) / 36 |> real

nums_T_L36 = Int64[]
for ix in 1:10000
    σs = perfect_sampling(ψs_T_L36[end])
    push!(nums_T_L36, num_domain_wall(σs))
end
histogram(nums_L36, fillcolor=:blue, alpha=0.5, bins = 1:36)
histogram!(nums_T_L36, fillcolor=:red, alpha=0.5, bins = 1:36)

dot(ψs_T_L36[end], ψs_L36[end])