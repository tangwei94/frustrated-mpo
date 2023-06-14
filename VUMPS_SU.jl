A = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^3*ℂ^3*ℂ^3*ℂ^3) 
A[1, 1, 3, 3, 3] = A[1, 3, 1, 3, 3] = A[1, 3, 3, 1, 3] = A[1, 3, 3, 3, 1] = 1
A[2, 2, 3, 3, 3] = A[2, 3, 2, 3, 3] = A[2, 3, 3, 2, 3] = A[2, 3, 3, 3, 2] = 2

B = TensorMap(ComplexF64[0 1 0 ; -1 0 0 ; 0 0 1], ℂ^3, ℂ^3)

@tensor Afull[-1; -2 -3 -4 -5] := A[-1; -2 -3 1 2] * B[1; -4] * B[2; -5]

χ = 3
δ1 = isomorphism(ℂ^(χ^2), (ℂ^χ)'*ℂ^χ)
δ1 = permute(δ1, (1, 2), (3, ))

δ2 = isomorphism((ℂ^(χ^2))', (ℂ^χ)'*ℂ^χ)
δ2 = permute(δ2, (2, ), (3, 1))

@tensor T[-1 -2; -3 -4] := Afull[9; 4 2 6 8] * Afull'[3 1 5 7; 9] * δ1[-1 2 ; 1] * δ1[-2 4; 3] * δ2[6; 5 -3] * δ2[8; 7 -4];
@tensor W[-1 ; -2] := B[4; 2] * B'[1; 3] * δ1[-1 2 ; 1] * δ2[3; 4 -2]

𝕋 = DenseMPO([T])
𝕋 = changebonds(𝕋, SvdCut(truncerr(1e-14)));
𝕎 = DenseMPO([add_util_leg(W)])

𝕊 = 𝕎 * 𝕋

# verify that 𝕊 is hermitian
S = 𝕊.opp[1]
Sdag = mpotensor_dag(S)
𝕊dag = DenseMPO([Sdag])
ϕ1 = convert(InfiniteMPS, 𝕊)
ϕ2 = convert(InfiniteMPS, 𝕊dag)
dot(ϕ1, ϕ2)

# "bidirectional" VUMPS
function optim1(ψ, maxiter, tol=1e-12)
    ix = 1
    while true
        envS = environments(ψ, 𝕊)
        ER = rightenv(envS, 1, ψ)
        EL = leftenv(envS, 1, ψ)
        EL = permute(EL, (1, ), (2, 3))

        envW = environments(ψ, 𝕎)
        ρR = rightenv(envW, 1, ψ)
        ρR = isomorphism(domain(ρR), codomain(ρR)) * ρR
        ρL = leftenv(envW, 1, ψ)
        ρL = isomorphism(domain(ρL), codomain(ρL)) * ρL

        invρR = inv(ρR + 1e-11 * id(domain(ρR)))
        invρL = inv(ρL + 1e-11 * id(domain(ρL)))

        λ = dot(ψ, 𝕊, ψ) / dot(ψ, 𝕎, ψ)

        function HAC(AC)
            @tensor newAC[-1 -2; -3] := invρL[-1; 1] * EL[1; 3 2] * AC[2 4 ; 5] * 𝕋.opp[1][3 -2; 4 6] * ER[5 6; 7] * invρR[7; -3]
        end
        function HC(C)
            @tensor newC[-1; -2] := invρL[-1; 1] * EL[1; 4 2] * C[2; 3] * ER[3 4; 5] * invρR[5; -2]
        end

        λAC, newAC = eigsolve(HAC, ψ.AC[1], 1, :LM, Arnoldi(tol=1e-10)) 
        λC, newC = eigsolve(HC, ψ.CR[1], 1, :LM, Arnoldi(tol=1e-10)) 

        function calculate_ALR(AC, C)
            UAC_l, RAC = leftorth(AC; alg=Polar())
            UC_l, RC = leftorth(C; alg=Polar())

            LAC, UAC_r = rightorth(permute(AC, (1,), (2,3)); alg=Polar())
            LC, UC_r = rightorth(C; alg=Polar())

            AL = UAC_l * UC_l'
            AR = permute(UC_r' * UAC_r, (1, 2), (3,))

            ϵL = norm(RAC - RC)
            ϵR = norm(LAC - LC)
            #ϵL = norm(AC - AL * C)
            #ϵR = norm(permute(AC, (1,), (2, 3)) - C * permute(AR, (1,), (2, 3)))

            return AL, AR, ϵL, ϵR   
        end

        newAL, newAR, ϵL, ϵR = calculate_ALR(newAC[1], newC[1])

        newψ = InfiniteMPS([newAL])
        ψ = newψ

        @show ix, real(λ), ϵL
        (ix == maxiter || ϵL < tol) && (return ψ, real(λ))
        ix += 1
    end
end

# ordinary VUMPS
function optim2(ψ, maxiter, tol=1e-12)
    ix = 1
    while true
        envT = environments(ψ, 𝕋)
        ER = rightenv(envT, 1, ψ)
        EL = leftenv(envT, 1, ψ)
        EL = permute(EL, (1, ), (2, 3))
    
        envW = environments(ψ, 𝕎)
        ρR = rightenv(envW, 1, ψ)
        ρR = isomorphism(domain(ρR), codomain(ρR)) * ρR
        ρL = leftenv(envW, 1, ψ)
        ρL = isomorphism(domain(ρL), codomain(ρL)) * ρL
    
        invρR = id(domain(ρR))
        invρL = id(domain(ρL))
        
        λ1 = dot(ψ, 𝕊, ψ) / dot(ψ, 𝕎, ψ)
        λ2 = dot(ψ, 𝕋, ψ) 
    
        function HAC(AC)
            @tensor newAC[-1 -2; -3] := invρL[-1; 1] * EL[1; 3 2] * AC[2 4 ; 5] * 𝕋.opp[1][3 -2; 4 6] * ER[5 6; 7] * invρR[7; -3]
        end
        function HC(C)
            @tensor newC[-1; -2] := invρL[-1; 1] * EL[1; 4 2] * C[2; 3] * ER[3 4; 5] * invρR[5; -2]
        end
    
        λAC, newAC = eigsolve(HAC, ψ.AC[1], 1, :LM, Arnoldi(tol=1e-10)) 
        λC, newC = eigsolve(HC, ψ.CR[1], 1, :LM, Arnoldi(tol=1e-10))
    
        function calculate_ALR(AC, C)
            UAC_l, RAC = leftorth(AC; alg=Polar())
            UC_l, RC = leftorth(C; alg=Polar())
        
            LAC, UAC_r = rightorth(permute(AC, (1,), (2,3)); alg=Polar())
            LC, UC_r = rightorth(C; alg=Polar())
        
            AL = UAC_l * UC_l'
            AR = permute(UC_r' * UAC_r, (1, 2), (3,))
        
            ϵL = norm(RAC - RC)
            ϵR = norm(LAC - LC)
            #ϵL = norm(AC - AL * C)
            #ϵR = norm(permute(AC, (1,), (2, 3)) - C * permute(AR, (1,), (2, 3)))
        
            return AL, AR, ϵL, ϵR   
        end
    
        newAL, newAR, ϵL, ϵR = calculate_ALR(newAC[1], newC[1])
    
        newψ = InfiniteMPS([newAL])
        ψ = newψ
    
        @show ix, real(λ1), ϵL

        (ix == maxiter || ϵL < tol) && (return ψ, real(λ1))
        ix += 1
    end
end

ψ = InfiniteMPS([ℂ^9], [ℂ^18]);
ψ0, λ0 = optim2(ψ, 10);

ψ1, λ1 = optim1(ψ0, 100);
ψ2, λ2 = optim2(ψ0, 100);
@show λ1, λ2, λ3
l1 = λ1 
l2 = λ2 
l3 = λ3

function VOMPS_history()
    optim_alg1 = VUMPS(tol_galerkin=1e-9, maxiter=100) 
    ψp = InfiniteMPS([ℂ^9], [ℂ^9])

    λ = 1
    for ix in 1:100
        #ψp0 = changebonds(𝕋*ψp, SvdCut(truncdim(9)))
        ψp, _ = approximate(ψp, (𝕋, ψp), optim_alg1)
        ψpl = 𝕎 * ψp 
        λ = real(dot(ψpl, 𝕋, ψp) / dot(ψpl, ψp))
        printstyled("$(left_virtualspace(ψ1, 1)), $(ix), $(λ) \n"; color=:red)
    end
    return ψp, λ
end

ψp, λp = VOMPS_history();
@show λ1, λ2, λp