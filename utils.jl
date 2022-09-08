"""
    tesnor_triangular_AF_ising()
    
    tensor for frustrated MPO.
"""
function tensor_triangular_AF_ising()
    # frustrated
    # exact: 0.3230659669
    # ref: Phys. Rev. Res. 3, 013041 (2021)
    t = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
    p = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2)
    t[1, 1, 2] = 1
    t[1, 2, 1] = 1
    t[2, 1, 1] = 1
    t[2, 2, 1] = 1
    t[2, 1, 2] = 1
    t[1, 2, 2] = 1
    p[1, 1, 1] = 1
    p[2, 2, 2] = 1
    T = t*p
    return T
end

"""
    tensor_triangular_AF_ising_T()
    
    frustrated MPO. The other direction.
"""
function tensor_triangular_AF_ising_T()
    # frustrated
    # exact: 0.3230659669
    # ref: Phys. Rev. Res. 3, 013041 (2021)
    T0 = tensor_triangular_AF_ising() 
    Tdata = reshape(T0.data, (2, 2, 2, 2))
    Tdata = permutedims(Tdata, (1, 3, 2, 4))
    return TensorMap(Tdata, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
end

"""
    tensor_triangular_AF_ising_alternative()

    non-frustrated MPO.
"""
function tensor_triangular_AF_ising_alternative()
    # non-frustrated 
    # exact: 0.3230659669
    # ref: Phys. Rev. Res. 3, 013041 (2021)
    t = TensorMap(zeros, ComplexF64, ℂ^4*ℂ^4, ℂ^4)
    t[1, 2, 3] = 1
    t[3, 1, 2] = 1
    t[2, 3, 1] = 1
    t[3, 2, 4] = 1
    t[2, 4, 3] = 1
    t[4, 3, 2] = 1
    @tensor T[-1, -2; -3, -4] := t'[-2, 1, -4] * t[-1, 1, -3]
    return T
end

"""
    mpo_gen(L::Int, mpo_choice::Symbol, boundary_condition::Symbol)

    Generate the MPO for transfer matrix 𝕋. `L` is the length of the system.
    mpo_choice can be chosen among `:frstr`, `:nonfrstr`, `:frstrT`
    boundary_condition `:pbc` or `:obc`
"""
function mpo_gen(L::Int, mpo_choice::Symbol, boundary_condition::Symbol)
    if mpo_choice == :frstr
        T = tensor_triangular_AF_ising();
        Dvir = 2;
    elseif mpo_choice == :nonfrstr 
        T = tensor_triangular_AF_ising_alternative(); 
        Dvir = 4;
    elseif mpo_choice == :frstrT 
        T = tensor_triangular_AF_ising_T();
        Dvir = 2;
    end

    if boundary_condition == :pbc 
        return periodic_boundary_conditions(DenseMPO(T), L)
    elseif boundary_condition == :obc 
        bT = TensorMap(fill(1.0, (Dvir, 1)), ℂ^Dvir, ℂ^1)
        𝕋 = fill(T, L)
        𝕋[1] = (@tensor T1[-1, -2; -3, -4] := bT'[-1, 1] * T[1, -2, -3, -4])
        𝕋[end] = (@tensor Tend[-1, -2; -3, -4] := T[-1, -2, -3, 1] * bT[1, -4]) 
        return DenseMPO(𝕋)
    end
end

"""
    perfect_sampling(ψ::FiniteMPS)

    perfect sampling for finite MPS.
    Ref: A. J. Ferris, G. Vidal, Phys. Rev. B 85, 165146 (2012) 
"""
function perfect_sampling(ψ::FiniteMPS)
    EL = id(ℂ^1)
    σs = Int64[]
    ξs = rand(length(ψ))
    ph_dim = dim(space(ψ, 1))
    P1 = 1
    for ix in 1:length(ψ)
        AR = ψ.AR[ix]
        @tensor ρx[-1; -2] := EL[3, 2] * AR[2, -1, 1] * AR'[1, 3, -2]

        cumPs = cumsum(real.(diag(ρx.data)))
        cumPs = cumPs ./ P1
    
        σ = 1 + sum(ξs[ix] .> cumPs)
        Dσ = TensorMap(zeros, ℂ^ph_dim, ℂ^ph_dim)
        P1 = real(ρx.data[σ, σ])

        Dσ.data[σ, σ] = 1
        push!(σs, σ)

        @tensor EL[-1; -2] := EL[2, 1] * AR[1, 3, -2] * AR'[-1, 2, 4] * Dσ[4, 3]        
    end
    return σs
end

"""
    num_domain_wall(σs::Vector{Int64})

    number of domain walls for a certain configuration
"""
function num_domain_wall(σs::Vector{Int64})
    N = length(σs)
    return sum(abs.(σs[[2:N; 1]] .- σs))
end

"""
    num_nfr_domain_wall(σs::Vector{Int64})

    number of domain walls for a certain configuration
"""
function num_nfr_domain_wall(σs::Vector{Int64})
    N = length(σs)
    return sum(σs .== 2) + sum(σs .== 3)
end

function accumulate_domain_wall_loc!(domain_wall_locs::Array{Int64, 1}, σs)
    domain_wall_locs[1:end-1] += abs.(σs[2:end] - σs[1:end-1])
end

function entanglement_entropy(ψ::FiniteMPS, loc::Int)
    spect = entanglement_spectrum(ψ, loc)
    return sum(-spect.^2 .* log.(spect.^2))
end

"""
    power_projection(𝕋::DenseMPO, χs::Vector{<:Int}; Npower=100, spect_shifting=0.2, spect_rotation=0, filename="temp.jld")   

    obtain fixed point MPS using power method
"""
function power_projection(𝕋::DenseMPO, χs::Vector{<:Int}; Npower=100, spect_shifting=0.2, spect_rotation=0, filename="temp")
    ph_space = space(𝕋.opp[2], 2)
    L = length(𝕋)

    fs = ComplexF64[] # free energy. although may not accurate
    vars = Float64[] # nonhermtian variance 
    diffs = Float64[] # fidelity with respect to previous step
    ψms = [] # optimized MPS for each χ

    ψm = FiniteMPS(L, ph_space, ℂ^χs[1]);
    for χ in χs
        varm = 1
        ψ = copy(ψm)
        Tψ = normalize(𝕋*ψ); 
        for ix in 1:Npower
            ψ1 = changebonds(exp(im*spect_rotation)*Tψ + spect_shifting*ψ, SvdCut(trscheme=truncdim(χ)));
            normalize!(ψ1)
            diff = 2*log(norm(dot(ψ, ψ1)))
            ψ = copy(ψ1)

            Tψ = 𝕋*ψ 
            f = log(dot(ψ, Tψ)) / L
            normalize!(Tψ)

            var = -2*log(norm(dot(ψ, Tψ))) / L |> real
            if abs(var) < varm 
                varm = abs(var) 
                ψm = copy(ψ)
            end

            push!(fs, f)
            push!(vars, abs(var))
            push!(diffs, diff)
        end
        push!(ψms, ψm)
        @show L, χ, minimum(vars)
    end
    @save filename*"L$(L).jld" {compress=true} fs vars diffs ψms 
    return fs, vars, diffs, ψms  
end