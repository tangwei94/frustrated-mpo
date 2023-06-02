using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

if length(ARGS) == 0
    mpo_choices = [:frstr, :frstrT]
else 
    mpo_choices = [:nonfrstr, :nonfrstrT]
end
mpo_choices = [:nonfrstr, :nonfrstrT]

boundary_condition = :inf
filename1 = filename_gen(mpo_choices[1], boundary_condition)
filename2 = filename_gen(mpo_choices[2], boundary_condition)

#χs = [2, 4, 8]
χs = [2, 4, 8, 16, 32, 64]

𝕋1 = mpo_gen(1, mpo_choices[1], boundary_condition); 
f1, vars1, diffs1, ψms1 = power_projection(𝕋1, χs; Npower=30, filename=filename1);

𝕋2 = mpo_gen(1, mpo_choices[2], boundary_condition); 
f2, vars2, diffs2, ψms2 = power_projection(𝕋2, χs; Npower=30, filename=filename2);

#plot()
#plot!(vars1, yaxis=:log)
#plot!(vars2, yaxis=:log)

@load filename1*"L1.jld" ψms 
ψms1 = copy(ψms)
@load filename2*"L1.jld" ψms 
ψms2 = copy(ψms)

dot(ψms1[end], 𝕋1, ψms1[end])

errs = Float64[]
for (ψL, ψR) in zip(ψms2, ψms1)
    f = dot(ψL, 𝕋1, ψR) / dot(ψL, ψR) |> norm |> log;
    @show f, (f-exact_free_energy) / exact_free_energy
    push!(errs, (f-exact_free_energy) / exact_free_energy)
end

gr()
scatter(dim.(_firstspace.([ψ.AL[1] for ψ in ψms1]))[1:8], abs.(errs)[1:8], color=:blue, alpha=0.5, xaxis=:log, yaxis=:log)
xlabel!("χ")
ylabel!("error in F")

@load filename_gen(:frstr, :inf) * "L1.jld" vars
vars_R = copy(vars)
@load filename_gen(:frstrT, :inf) * "L1.jld" vars
vars_L = copy(vars)
@load filename_gen(:frstr, :inf) * "L1.jld" ψms
ψms_R = copy(ψms)
@load filename_gen(:frstrT, :inf) * "L1.jld" ψms
ψms_L = copy(ψms)

plot(vars_R, yaxis=:log)
plot!(vars_L, yaxis=:log)

fs = [dot(ψL, 𝕋, ψR) / dot(ψL, ψR) for (ψL, ψR) in zip(ψms_L, ψms_R)]
norm.(log.(fs))
@show exact_free_energy