using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using CairoMakie
using FiniteDifferences, Optim

#function entanglement_entropy(ψ::InfiniteMPS)
#    spect = entanglement_spectrum(ψ)
#    return sum(-spect.^2 .* log.(spect.^2))
#end

τs = -1:0.01:1

#measure physical quantity \sum_i \sigma_i^z
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
function new_EE(ψ::InfiniteMPS, τ::Real)
    P = exp(-τ * σz)
    ℙ = DenseMPO([add_util_leg(P)])
    ϕ = ℙ * ψ 
    return real(entropy(ϕ)[1])
end

ψ1 = InfiniteMPS([ℂ^2], [ℂ^2])
EE1s = new_EE.(Ref(ψ1), τs)
ψ2 = InfiniteMPS([ℂ^2], [ℂ^2])
EE2s = new_EE.(Ref(ψ2), τs)
ψ3 = InfiniteMPS([ℂ^2], [ℂ^2])
EE3s = new_EE.(Ref(ψ3), τs)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"S(\mathrm{exp}(-\tau M) |\psi\rangle)", yscale=log10)
scatter1 = scatter!(ax1, τs, abs.(EE1s) .+ 1e-16, marker=:circle, markersize=5, label=L"\psi_1")
scatter1 = scatter!(ax1, τs, abs.(EE2s) .+ 1e-16, marker=:circle, markersize=5, label=L"\psi_2")
scatter1 = scatter!(ax1, τs, abs.(EE3s) .+ 1e-16, marker=:circle, markersize=5, label=L"\psi_3")
axislegend(ax1; position=:rb)
@show fig
save("local-gauge-random-iMPS.pdf", fig)

function maximal_tau(ψ::InfiniteMPS)
    _f(τ) = -new_EE(ψ, τ[1])
    _g(τ) = FiniteDifferences.central_fdm(5, 1)(_f, τ[1])

    res = optimize(_f, _g, [0.0], GradientDescent(); inplace=false)
    return Optim.minimizer(res)[1]
end

ψ = InfiniteMPS([ℂ^2], [ℂ^2])
maximal_tau(ψ)
expectation_value(ψ, σz, 1)

τms = Float64[] 
zs = Float64[]
for ix in 1:1000
    ψ = InfiniteMPS([ℂ^2], [ℂ^2])
    push!(τms, maximal_tau(ψ))
    push!(zs, real(expectation_value(ψ, σz, 1)))
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"\langle \sigma_z \rangle", ylabel=L"\tau_0")
scatter1 = scatter!(ax1, zs, τms, marker=:circle, markersize=10)
line1 = lines!(ax1, [minimum(zs), maximum(zs)], [minimum(zs), maximum(zs)] .* 0.5, marker=:circle, color=:red)
@show fig
save("local-gauge-random-iMPS-2.pdf", fig)