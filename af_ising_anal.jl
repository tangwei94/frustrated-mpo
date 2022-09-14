using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choice = :nonfrstr # :nonfrstr, :frstrT
boundary_condition = :obc # :pbc, :obc

filename = filename_gen(mpo_choice, boundary_condition)

Ls = [12, 18, 24, 30, 36, 48, 60, 72, 84]; # totally breaks at L=96 
χs = [4, 8, 12, 16, 20, 24, 28, 32];

vars_L = []
ψms_L = []
for L in Ls
    @load filename*"L$(L).jld" vars
    @load filename*"L$(L).jld" ψms
    push!(vars_L, vars)
    push!(ψms_L, ψms)
end

# varaince
plot(0,1)
for (L, vars) in zip(Ls, vars_L) 
    plot!(vars .+ 1e-16, yaxis=:log, alpha=0.75, label="L=$(L)")
end
ylabel!("variance")
xlabel!("power steps")
savefig(filename*"variance.pdf")
#current()

# half-chain entanglement vs L 
plot()
EEs = Float64[]
for (L, ψms) in zip(Ls, ψms_L)
    ψm = ψms[end] 
    EE = entanglement_entropy(ψm, L ÷ 2)
    push!(EEs, EE)     
end
scatter!(Ls, EEs)
xlabel!("L")
ylabel!("half chain EE")
M = fill(1.0, (length(Ls)-length(Ls)÷2+1, 2))
M[:, 2] = log.(Ls[(length(Ls)÷2):end])
fits = (M' * M) \ (M' * EEs[(length(Ls)÷2):end]) # linear fitting
@show fits
plot!(Ls, fits[1] .+ fits[2] .* log.(Ls))
xaxis!(:log)
savefig(filename*"halfchainEE-vs-L.pdf")

plot()
for (L, ψms) in zip(Ls, ψms_L)
    ψm = ψms[end] 
    ES = entanglement_spectrum(ψm, L ÷ 2)
    scatter!(fill(L, length(ES)), ES, alpha=0.5, yaxis=:log, legend=false)
end
xlabel!("L")
ylabel!("half chain ES")
savefig(filename*"halfchainES-vs-L.pdf")

# half-chain entanglement vs χ 
plot()
EEs = Float64[]
for (χ, ψm) in zip(χs, ψms_L[end])
    EE = entanglement_entropy(ψm, Ls[end] ÷ 2)
    push!(EEs, EE)    
end
plot!(χs, EEs)
xlabel!("χ")
ylabel!("half chain EE")
savefig(filename*"halfchainEE-vs-chi.pdf")

plot()
for (χ, ψm) in zip(χs, ψms_L[end])
    ES = entanglement_spectrum(ψm, Ls[end] ÷ 2)
    scatter!(fill(χ, length(ES)), ES, alpha=0.5, yaxis=:log, legend=false)
end
xlabel!("χ")
ylabel!("half chain ES")
savefig(filename*"halfchainES-vs-chi.pdf")

# entanglement vs subsystem L
plot()
EEs = Float64[]
ψms = ψms_L[end]
for Lsub in 2:Ls[end]-1
    EE = entanglement_entropy(ψms[end], Lsub)
    push!(EEs, EE)
end
plot!(Vector(2:Ls[end]-1), EEs)
xlabel!("L of subsystem")
ylabel!("EE")
savefig(filename*"EE-vs-subL.pdf")

plot()
for Lsub in 2:Ls[end]-1
    ES = entanglement_spectrum(ψms[end], Lsub)
    scatter!(fill(Lsub, length(ES)), ES, alpha=0.5, yaxis=:log, legend=false)
end
xlabel!("L of sybsystem")
ylabel!("ES")
savefig(filename*"ES-vs-subL.pdf")

# correlation function 
if mpo_choice in [:frstr, :frstrT]
    σz = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    σz.data[1, 1] = 1
    σz.data[2, 2] = -1
    σ0 = id(ℂ^2)
elseif mpo_choice == :nonfrstr
    σz = TensorMap(zeros, ComplexF64, ℂ^4, ℂ^4)
    σz.data[1, 1] = 1
    σz.data[2, 2] = -1
    σz.data[3, 3] = 1
    σz.data[4, 4] = -1
    σ0 = id(ℂ^4)
end
corrs = Float64[]
correlator = [add_util_leg(σz), add_util_leg(σz)]

#push!(corrs, abs(expectation_value(ψms[end], correlator, Ls[end] ÷ 2)))
for ix in 1:Ls[end]÷3
    insert!(correlator, 2, add_util_leg(σ0))
    push!(corrs, abs(expectation_value(ψms[end], correlator, Ls[end]÷2-ix)))
    insert!(correlator, 2, add_util_leg(σ0))
    push!(corrs, abs(expectation_value(ψms[end], correlator, Ls[end]÷2-ix)))
end
plot(Vector(1:Ls[end]÷3*2), corrs, xaxis=:log, yaxis=:log)
xlabel!("seperation")
ylabel!("correlation")
savefig(filename*"corr-vs-distance.pdf")


#Atensors = MPSKit.MPSTensor{ComplexSpace}[]
#for ix in 1:18
#    push!(Atensors, ϕ.AL[ix])
#end
#Atensors[18] = TensorMap(reshape(Atensors[18].data, (24, 2, 24))[:, :, 1], ℂ^24*ℂ^2, ℂ^1)
#ϕA1 = FiniteMPS(Atensors)
#for ix in 1:100
#    sA = perfect_sampling(ϕA1)
#    @show sum(abs.(sA[2:end] - sA[1:end-1]))
#end