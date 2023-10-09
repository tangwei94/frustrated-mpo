# exact free energy at T=0 for triangular ising
const exact_free_energy = 0.3230659669 

# copied from MPSKit.jl/src/states/abstractmps.jl
const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}

# copied from MPSKit.jl/src/utility.jl
_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))
    
"""
    tensor_trivial()

    construct a nonhermitian MPO in the form P 𝕋 P^{-1}, where 𝕋 is a hermitian MPO.
"""
function tensor_trivial(β::Real, ϵ::Number)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    δ[1, 1, 1, 1] = δ[2, 2, 2, 2] = 1

    t = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    t[1, 1] = t[2, 2] = exp(β)
    t[2, 1] = t[1, 2] = exp(-β)
    U, S, V = tsvd(t)
    Us, sV = U * sqrt(S), sqrt(S) * V

    @tensor T[-1, -2; -3, -4] := sV[-1, 1] * sV[-2, 2] * Us[3, -3] * Us[4, -4] * δ[1, 2, 3, 4]

    Pdat = [1 2 ; 4 2] * ϵ + Matrix{ComplexF64}(I, 2, 2)
    Pinvdat = inv(Pdat)

    P = TensorMap(Pdat, ℂ^1*ℂ^2, ℂ^2*ℂ^1)
    Pinv = TensorMap(Pinvdat, ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    Pdag = TensorMap(Matrix(Pdat'), ℂ^1*ℂ^2, ℂ^2*ℂ^1)
    Pdaginv = TensorMap(Matrix(Pinvdat'), ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    return T, P, Pinv, Pdag, Pdaginv
end

"""
    tensor_square_ising(β::Real)

    return the local tensor for FM sqaure lattice ising. 
"""
function tensor_square_ising(β::Real)
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end

"""
    tensor_triangular_AF_ising()
    
    tensor for frustrated MPO.
"""
function tensor_triangular_AF_ising()
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
    tensor_percolation(p1::Float64, p2::Float64)

    Domany-Kinzel model. See E. Domany, W. Kinzel, Phys. Rev. Lett. 53, 311–314
(1984). A clearer description of the model can be found in H. Hinrichsen, Adv. Phys. 49, 815–958 (2000). 
"""
function tensor_percolation(p1::Real, p2::Real)
    δ = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2)
    S = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)

    δ[1, 1, 1] = 1
    δ[2, 2, 2] = 1

    S[1, 1, 1] = 1
    S[1, 2, 1] = 0
    S[1, 2, 2] = S[2, 2, 1] = p1 
    S[1, 1, 2] = S[2, 1, 1] = 1 - p1
    S[2, 2, 2] = p2 
    S[2, 1, 2] = 1 - p2

    @tensor A[-1 -2; -3 -4] := S[-1 -2; 1] * δ[1; -3 -4]
    @tensor B[-1 -2; -3 -4] := δ[-1; -3 1] * S[1 -2; -4]

    t_fuse = isomorphism(ℂ^4, ℂ^2*ℂ^2)
    @tensor T[-1 -2; -3 -4] := t_fuse[-1; 1 2] * A[2 3; -3 5] * B[1 -2; 3 4] * t_fuse'[4 5; -4]

    return T
end 

"""
    tensor_triangular_AF_ising_T()
    
    frustrated MPO. The other direction.
"""
function tensor_triangular_AF_ising_T()
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
    tensor_triangular_AF_ising_alternative_T()

    non-frustrated MPO. the other direction.
"""
function tensor_triangular_AF_ising_alternative_T()
    T0 = tensor_triangular_AF_ising_alternative() 
    Tdata = reshape(T0.data, (4, 4, 4, 4))
    Tdata = permutedims(Tdata, (1, 3, 2, 4))
    return TensorMap(Tdata, ℂ^4*ℂ^4, ℂ^4*ℂ^4)
end

"""
    tensor_triangular_AF_ising_adapted()

    non-frustrated MPO. gauge the physical dim to be 2.
"""
function tensor_triangular_AF_ising_adapted() 

    p = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
    m = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^4*ℂ^2)

    p[1, 1, 1] = p[2, 2, 2] = 1
    #m[1, 1, 1] = m[1, 2, 2] = m[2, 3, 1] = m[2, 4, 2] = 1
    m[1, 1, 1] = m[2, 2, 1] = m[1, 3, 2] = m[2, 4, 2] = 1
    #@tensor A[-1, -2; -3, -4] := p[1, -2, -4] * m[-1, -3, 1]

    δ = isomorphism(ℂ^16, (ℂ^2)'*ℂ^4*ℂ^2)
    δ = permute(δ, (1, 2), (3, 4))

    T0 = tensor_triangular_AF_ising_alternative()
    #@tensor T1[-1, -2; -3, -4] := δ[-1, 5, 4, 3] * A[3, -2, 1, 6] * T0[4, 1, 2, 7] * A'[2, 8, 5, -3] * δ'[7, 6, -4, 8]
    #@tensor T1[-1 -2; -3 -4] := δ[-1 3 ; 2 1] * m[1 ; 4 6] * T0[2 4 ; 5 7] * m'[5 8; 3] * p[6 -2; 9] * p'[10 8 -3] * δ'[7 9 ; -4 10]
    @tensor T1[-1 -2; -3 -4] := δ[-1 3 ; 2 1] * p[1 -2 ; 4] * m[4 ; 5 7] * T0[2 5 ; 6 8] * m'[6 9 ; 3] * p'[10 9; -3] * δ'[8 7 ; -4 10]

    U, S, V, _ = tsvd(permute(T1, (1, ), (2, 3, 4)), trunc = truncerr(1e-12))
    @show S
    V = permute(V, (1, 2), (3, 4))
    @tensor T[-1, -2; -3, -4] := S[-1, 1] * V[1, -2, -3, 2] * U[2, -4]
    return T
end

"""
    mpotensor_dag(T::MPOTensor)

    Generate the hermitian conjugate of a MPO.
"""
function mpotensor_dag(T::MPOTensor)
    T_data = reshape(T.data, (dims(codomain(T))..., dims(domain(T))...))
    Tdag_data = permutedims(conj.(T_data), (1, 3, 2, 4))
    
    return TensorMap(Tdag_data, space(T))
end

"""
    mpo_gen(L::Int, mpo_choice::Symbol, boundary_condition::Symbol)

    Generate the MPO for transfer matrix 𝕋. `L` is the length of the system.
    mpo_choice can be chosen among `:frstr`, `:nonfrstr`, `:frstrT`, `:nonfrstrT`
    boundary_condition `:pbc` or `:obc`
    use L=1 or boundary_condition `:inf` for infinite system
"""
function mpo_gen(L::Int, mpo_choice::Symbol, boundary_condition::Symbol)
    if mpo_choice == :frstr
        T = tensor_triangular_AF_ising();
    elseif mpo_choice == :nonfrstr 
        T = tensor_triangular_AF_ising_alternative(); 
    elseif mpo_choice == :frstrT 
        T = tensor_triangular_AF_ising_T();
    elseif mpo_choice == :nonfrstrT 
        T = tensor_triangular_AF_ising_alternative_T(); 
    elseif mpo_choice == :nonfrstr_adapted
        T = tensor_triangular_AF_ising_adapted()
    end
    return mpo_gen(L, T, boundary_condition)
end

"""
    mpo_gen(L::Int, T::MPOTensor, boundary_condition::Symbol)

    Generate the MPO for transfer matrix 𝕋. 
    `L` is the length of the system.
    boundary_condition `:pbc` or `:obc`
    use L=1 or boundary_condition `:inf` for infinite system
"""
function mpo_gen(L::Int, T::MPOTensor, boundary_condition::Symbol)

    Dvir = dim(_firstspace(T))

    if L == 1 || boundary_condition == :inf # infinite
        return DenseMPO([T])
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
    convert_to_mat(𝕋::DenseMPO)

    convert a (pressumed to be finite) MPO into a big matrix. 
    Be careful about the MPO's length!
"""
function convert_to_mat(𝕋::DenseMPO)

    L = length(𝕋)

    ncon_contraction_order = [[ix, -2*ix+1, -2*ix, ix+1] for ix in 1:L] 
    ncon_contraction_order[end][end] = 1
    permutation_orders = Tuple(2 .* (1:L) .- 1), Tuple(2 .* (1:L))

    𝕋mat = permute(ncon(𝕋.opp, ncon_contraction_order), permutation_orders...)   

    return 𝕋mat
end


"""
    circular_mps(A, L, ϵ=1e-8)

    Construct a uniform circular MPS using the given MPS tensor `A`. Then convert this this circular MPS to an open-boundary MPS.
"""
function circular_mps(A::MPSKit.GenericMPSTensor, L::Int, ϵ::Real=1e-8)
    sp = MPSKit._firstspace(A)
    B = permute(isomorphism(ℂ^1*sp', sp), (1,), (2, 3,))
    t = id(sp)
    M = B
    As = map(1:L) do ix
        if ix < L
            @tensor A1[-1 -2 ; -3 -4] := M[-1 ; 1 2] * A[1 -2 ; -3] * t[2; -4]
            L1, M = leftorth(A1, (1, 2), (3, 4))
        else
            @tensor A1[-1 -2 ; -3] := M[-1; 1 2] * A[1 -2; 3] * t[2; 4] * B'[3 4 ; -3]
            L1, M = leftorth(A1, (1, 2), (3,))
        end
        return L1
    end

    M = id(ℂ^1)
    ix = 1
    for ix in L:-1:1
        A1 = As[ix] * M
        M, S, R1 = tsvd(A1, (1,), (2, 3), trunc=truncerr(ϵ))
        As[ix] = permute(R1, (1, 2), (3,))
        M = M * S
    end

    return FiniteMPS(As)
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
    num_domain_wall(σs::Vector{Int64}, mpo_choice::Symbol, boundary_condition::Symbol)

    number of domain walls for a certain configuration
"""
function num_domain_wall(σs::Vector{Int64}, mpo_choice::Symbol, boundary_condition::Symbol)
    L = length(σs)
    if mpo_choice in [:frstr, :frstrT]
        if boundary_condition == :pbc
            return sum(abs.(σs[[2:L; 1]] .- σs))
        elseif boundary_condition == :obc 
            return sum(abs.(σs[2:L] .- σs[1:L-1]))
        end
    elseif mpo_choice in [:nonfrstr, :nonfrstrT] 
        if boundary_condition == :pbc
            return sum(σs .== 2) + sum(σs .== 3)
        elseif boundary_condition == :obc
            @warn "not checked" 
            return sum(σs .== 2) + sum(σs .== 3)
        end
    end
end

"""
    sample_n_domain_wall(ψ::finitemps, mpo_choice::symbol, boundary_condition::symbol; ntotal=1000)
"""
function sample_n_domain_wall(ψ::FiniteMPS, mpo_choice::Symbol, boundary_condition::Symbol; Ntotal=1000)
    nums = Int64[]
    for ix in 1:Ntotal
        σs = perfect_sampling(ψ)
        push!(nums, num_domain_wall(σs, mpo_choice, boundary_condition))
    end
    return nums
end

function accumulate_domain_wall_loc!(domain_wall_locs::Array{Int64, 1}, σs)
    domain_wall_locs[1:end-1] += abs.(σs[2:end] - σs[1:end-1])
end

function entanglement_entropy(ψ::FiniteMPS, loc::Int)
    spect = entanglement_spectrum(ψ, loc)
    return sum(-spect.^2 .* log.(spect.^2))
end

function nonherm_variance!(Tψ::MPSKit.AbstractMPS, ψ::MPSKit.AbstractMPS)
    normalize!(ψ)
    normalize!(Tψ)
    L = length(ψ)
    return -2*log(norm(dot(ψ, Tψ))) / L |> real
end

struct operation_scheme 
    spect_shift::Number
    spect_rotation::Number
    project_outL::Vector{<:FiniteMPS}
    project_outR::Vector{<:FiniteMPS}

    function operation_scheme(spect_shift::Number,
                           spect_rotation::Number, 
                           project_outL::Vector{<:FiniteMPS},
                           project_outR::Vector{<:FiniteMPS})

        project_outL_binormalized = FiniteMPS[]
        for (ϕL, ϕR) in zip(project_outL, project_outR)
            push!(project_outL_binormalized, ϕL * (1/dot(ϕR, ϕL)))
        end

        new(spect_shift, spect_rotation, project_outL_binormalized, project_outR)
        #new(spect_shift, spect_rotation, project_outL, project_outR)
    end
end

function (a::operation_scheme)(Tψ::FiniteMPS, ψ::FiniteMPS)
    ψ1 = exp(im*a.spect_rotation)*Tψ + a.spect_shift*ψ
    for (ϕL, ϕR) in zip(a.project_outL, a.project_outR)
        ψ1 = ψ1 - dot(ϕL, ψ1) * ϕR
        #ψ1 = ψ1 - dot(ϕL, ψ1) * ϕL
    end
    return ψ1
end

const gs_operation = operation_scheme(0.2, 0, FiniteMPS[], FiniteMPS[])
const no_operation = operation_scheme(0, 0, FiniteMPS[], FiniteMPS[])
const gs1_operation = operation_scheme(0.4, 2*pi/3, FiniteMPS[], FiniteMPS[])
const gs2_operation = operation_scheme(0.4, -2*pi/3, FiniteMPS[], FiniteMPS[])

"""
    function power_projection(𝕋::DenseMPO, χs::Vector{<:Int}; Npower=100, operation=gs_operation, filename="temp")

    obtain fixed point MPS using power method
"""
function power_projection(𝕋::DenseMPO, χs::Vector{<:Int}; Npower=100, operation=gs_operation, filename="temp")
    ph_space = space(𝕋.opp[2], 2)
    L = length(𝕋)

    fs = ComplexF64[] # free energy. although may not accurate
    vars = Float64[] # nonhermtian variance 
    diffs = Float64[] # fidelity with respect to previous step
    ψms = [] # optimized MPS for each χ

    if L > 1
        ψm = FiniteMPS(L, ph_space, ℂ^χs[1]);
        for χ in χs
            varm = 1
            ψ = copy(ψm)
            Tψ = normalize(𝕋*ψ); 
            for ix in 1:Npower
                ψ1 = changebonds(operation(Tψ, ψ), SvdCut(trscheme=truncdim(χ)));
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
    elseif L == 1
        ψm = InfiniteMPS([ph_space], [ℂ^χs[1]]);
        for χ in χs
            varm = 1
            ψ = copy(ψm)
            Tψ = 𝕋*ψ;
            normalize!(Tψ) 
            for ix in 1:Npower
                ψ1 = changebonds(Tψ, SvdCut(trscheme=truncdim(χ)));
                normalize!(ψ1)
                diff = 2*log(norm(dot(ψ, ψ1)))
                ψ = copy(ψ1)

                f = log(dot(ψ, 𝕋, ψ)) # not accurate 

                Tψ = 𝕋*ψ 
                normalize!(Tψ)

                var = -2*log(norm(dot(ψ, Tψ))) |> real
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
    end
    @save filename*"L$(L).jld" {compress=true} fs vars diffs ψms 
    return fs, vars, diffs, ψms  
end

"""
    filename_gen(mpo_choice::Symbol, boundary_condition::Symbol)

    Generate the filename for the datafile.
"""
function filename_gen(mpo_choice::Symbol, boundary_condition::Symbol; more_info="")
    if mpo_choice == :frstr
        d_ph = 2;
        filename = "frustrated_";
    elseif mpo_choice == :nonfrstr 
        d_ph = 4;
        filename = "nonfrustrated_";
    elseif mpo_choice == :frstrT 
        d_ph = 2;
        filename = "frustrated_T_";
    elseif mpo_choice == :nonfrstrT 
        d_ph = 4;
        filename = "nonfrustrated_T_";
    end
    filename = filename * String(boundary_condition) * "_"

    return filename * more_info
end


function ChainRulesCore.rrule(::typeof(TensorKit.exp), K::TensorMap)
    W, UR = eig(K)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end

    expK = UR * exp(W) * UL

    function exp_pushback(f̄wd)
        ēxpK = f̄wd 
       
        K̄ = zero(K)

        if ēxpK != ZeroTangent()
            if W.data isa TensorKit.SortedVectorDict
                # TODO. symmetric tensor
                error("symmetric tensor. not implemented")
            end
            function coeff(a::Number, b::Number) 
                if a ≈ b
                    return exp(a)
                else 
                    return (exp(a) - exp(b)) / (a - b)
                end
            end
            M = UR' * ēxpK * UL'
            M1 = similar(M)
            copyto!(M1.data, M.data .* coeff.(Ws', conj.(Ws)))
                        K̄ += UL' * M1 * UR'# - tr(ēxpK * expK') * expK'
        end
        return NoTangent(), K̄
    end 
    return expK, exp_pushback
end

function get_transferL(AL1, AL2)
    function transferL(v)
        @tensor vo[-1; -2] := v[1; 2] * conj(AL2[1 3; -1]) * AL1[2 3; -2]
    end
    return transferL
end
function get_transferR(AR1, AR2)
    function transferR(v)
        @tensor vo[-1; -2] := v[1; 2] * conj(AR2[-2 3; 2]) * AR1[-1 3; 1]
    end
    return transferR
end
function importance_scattering_L(ψl, ψr)
    AL1, AL2 = ψl.AL[1], ψr.AL[1]
    C1, C2 = ψl.CR[1], ψr.CR[2]
    return importance_scattering_L(AL1, AL2, C1, C2)
end 
function importance_scattering_L(AL1, AL2, C1, C2)
    U1, S1, V1 = tsvd(C1)
    U2, S2, V2 = tsvd(C2)
    @tensor AL1_n[-1 -2 ; -3] := U1'[-1; 1] * AL1[1 -2; 2] * U1[2; -3]
    @tensor AL2_n[-1 -2 ; -3] := U2'[-1; 1] * AL2[1 -2; 2] * U2[2; -3]

    transferL = get_transferL(AL1_n, AL2_n)

    sp = domain(AL1_n)
    v0 = TensorMap(rand, ComplexF64, sp, sp)

    _, ρl, _ = eigsolve(transferL, v0, 1, :LM)
    ρl = ρl[1]
    ρl /= maximum(norm.(ρl.data))
    return ρl#, sqrt(S2) * ρl * sqrt(S1)
end
function rhoLR(ψ1, ψ2)
    sp = domain(ψ1.AL[1])
    v0 = TensorMap(rand, ComplexF64, sp, sp)

    C1, C2 = ψ1.CR[1], ψ2.CR[1]
    U1, S1, V1 = tsvd(C1)
    U2, S2, V2 = tsvd(C2)
    @tensor AL1[-1 -2 ; -3] := U1'[-1; 1] * ψ1.AL[1][1 -2; 2] * U1[2; -3]
    @tensor AL2[-1 -2 ; -3] := U2'[-1; 1] * ψ2.AL[1][1 -2; 2] * U2[2; -3]
    @tensor AR1[-1 -2 ; -3] := V1[-1; 1] * ψ1.AR[1][1 -2; 2] * V1'[2; -3]
    @tensor AR2[-1 -2 ; -3] := V2[-1; 1] * ψ2.AR[1][1 -2; 2] * V2'[2; -3]
    
    transferL = get_transferL(AL1, AL2)
    transferR = get_transferR(AR1, AR2)

    _, ρl, _ = eigsolve(transferL, v0, 1, :LM)
    _, ρr, _ = eigsolve(transferR, v0, 1, :LM)
    ρl = ρl[1]
    ρr = ρr[1]
    ρlr = sqrt(S1) * ρr * S2' * ρl * sqrt(S1) 
    return ρlr / tr(ρlr)
end

function mpo_ovlp(A1, A2)
    χ1 = dim(MPSKit._lastspace(A1))
    χ2 = dim(MPSKit._lastspace(A2))

    function mpo_transf(v)
        @tensor Tv[-1; -2] := A1[-1 3; 4 1] * conj(A2[-2 3; 4 2]) * v[1; 2]
        return Tv
    end

    v0 = TensorMap(rand, ComplexF64, ℂ^χ1, ℂ^χ2)
    return eigsolve(mpo_transf, v0, 1, :LM)
end

function mpo_ovlp1(𝕋1, 𝕋1dag)
    a1 = 𝕋1.opp[1]
    a2 = 𝕋1dag.opp[1]

    normality = mpo_ovlp(a1, a2)[1][1] * mpo_ovlp(a2, a1)[1][1] / mpo_ovlp(a1, a1)[1][1] / mpo_ovlp(a2, a2)[1][1]

    return normality
end