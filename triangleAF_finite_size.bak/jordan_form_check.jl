using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using JLD2
using Plots

include("utils.jl");

mpo_choice = :frstr # :nonfrstr, :frstrT
boundary_condition = :pbc # :pbc, :obc

round_digit8(x) = round(x, digits=8)

𝕋3 = mpo_gen(3, mpo_choice, boundary_condition)
@tensor Hmat3[-1, -2, -3; -4, -5, -6] := 
    𝕋3[1][1, -1, -4, 2] * 
    𝕋3[2][2, -2, -5, 3] * 
    𝕋3[3][3, -3, -6, 1]  

Λ, P = eigen(Hmat3.data)
round_digit8.(Λ)
round_digit8.(P' * Hmat3.data * P)

gsr = P[:, end]
gsl = inv(P)[end, :]

norm(gsl' * Hmat3.data - 3 * gsl')

gsl' * gsr
norm(gsl), norm(gsr)

U, S, V = svd(Hmat3.data - 3* Matrix(I, (8, 8)))
gs = V[:, 8]
#gs1 = P[:, 8]
#gs2 = P[:, 3]
#gs3 = P[:, 4]

round_digit8.(P' * P[:, end])

svd(Hmat3.data + 1 * Matrix(I, (8, 8)))

h3 = copy(Hmat3.data)
for ix in 1:10000
    q3, r3 = qr(h3)
    h3 = r3 * q3
end
@show round_digit8.(h3)

𝕋6 = mpo_gen(6, mpo_choice, boundary_condition)
@tensor Hmat6[-1, -2, -3, -4, -5, -6; -7, -8, -9, -10, -11, -12] := 
    𝕋6[1][1, -1, -7, 2] * 
    𝕋6[2][2, -2, -8, 3] * 
    𝕋6[3][3, -3, -9, 4] * 
    𝕋6[4][4, -4, -10, 5] * 
    𝕋6[5][5, -5, -11, 6] * 
    𝕋6[6][6, -6, -12, 1] 

Λ6, P6 = eigen(Hmat6.data)
round_digit8.(Λ6)
round_digit8.(P6' * Hmat6.data * P6)[end-8:end, end-8:end]

gs6r = P6[:, end]
gs6l = inv(P6)[end, :]


norm(Hmat6.data * gs6r - Λ6[end]  * gs6r)

gsl' * gsr
norm(gsl), norm(gsr)

norm.(P6' * gs6)[norm.(P6' * gs6) .> 1e-8]
Λ6[norm.(P6' * gs6) .> 1e-8]
Λ6[(real.(Λ6) .> 1e-8) .* (abs.(imag.(Λ6)) .< 1e-8)]

U6, S6, V6 = svd(Hmat6.data - Λ6[end, end]* Matrix(I, (64, 64)))

gs6 = V6[:, end]
Hmat6.data * gs6 - Λ6[end, end] * gs6
gs6l = U6[:, end]
norm(gs6l' * Hmat6.data - Λ6[end] * gs6l')
gs6l' * gs6

T6 = Hmat6.data
log(dot(T6 * gs6, T6 * gs6) * dot(gs6, gs6) / dot(gs6, T6, gs6) / dot(gs6, T6', gs6))

variance(T, x) = real(log(dot(T * x, T * x) * dot(x, x) / dot(x, T, x) / dot(x, T', x)))

# power method 
gs_power6 = rand(64)
variances = Float64[]
for ix in 1:100
    gs_power6 = normalize(T6 * gs_power6)  
    push!(variances, variance(T6, gs_power6))
end
plot(abs.(variances) .+ 1e-16, yaxis=:log)
variances

gs_power6_rot1 = rand(64)
variances6_rot1 = Float64[]
for ix in 1:300
    gs_power6_rot1 = normalize(exp(im*2*pi/3) * T6 * gs_power6_rot1) + 0.2*gs_power6_rot1  
    push!(variances6_rot1, variance(T6, gs_power6_rot1))
end
plot!(abs.(variances6_rot1) .+ 1e-16, yaxis=:log)

norm(dot(gs_power6, gs_power6_rot1)) / norm(gs_power6) / norm(gs_power6_rot1)

normalize(gs6) - normalize(gs_power6) 

𝕋9 = mpo_gen(9, mpo_choice, boundary_condition);
@tensor Hmat9[-1, -2, -3, -4, -5, -6, -7, -8, -9; -10, -11, -12, -13, -14, -15, -16, -17, -18] := 
    𝕋9[1][1, -1, -10, 2] * 
    𝕋9[2][2, -2, -11, 3] * 
    𝕋9[3][3, -3, -12, 4] * 
    𝕋9[4][4, -4, -13, 5] * 
    𝕋9[5][5, -5, -14, 6] * 
    𝕋9[6][6, -6, -15, 7] *  
    𝕋9[7][7, -7, -16, 8] *  
    𝕋9[8][8, -8, -17, 9] *  
    𝕋9[9][9, -9, -18, 1] ; 

Λ9, P9 = eigen(Hmat9.data);
T9 = Hmat9.data;
Λm9 = eigvals(Hmat9.data)[end]
U9, S9, V9 = svd(Hmat9.data - Λm9 * Matrix(I, (512, 512)))

gs9 = P9[:, end]

gs9 = V9[:, end]
norm(Hmat9.data * gs9 - Λ9[end, end] * gs9)
gs9l = U9[:, end]
norm(gs9l' * Hmat9.data - Λ9[end] * gs9l')
norm(gs9l' * gs9)

# power method 
gs_power9 = rand(512)
variances9 = Float64[]
for ix in 1:100
    gs_power9 = normalize(T9 * gs_power9)  
    push!(variances9, variance(T9, gs_power9))
end
plot(abs.(variances9) .+ 1e-16, yaxis=:log)

gs_power9_rot1 = rand(512)
variances9_rot1 = Float64[]
for ix in 1:200
    gs_power9_rot1 = normalize(exp(im*2*pi/3) * T9 * gs_power9_rot1) + 0.2*gs_power9_rot1  
    push!(variances9_rot1, variance(T9, gs_power9_rot1))
end
plot!(abs.(variances9_rot1) .+ 1e-16, yaxis=:log)

norm(dot(gs_power9, gs_power9_rot1)) / norm(gs_power9) / norm(gs_power9_rot1)

gs_power9_rot2 = rand(512)
variances9_rot2 = Float64[]
for ix in 1:200
    gs_power9_rot2 = normalize(exp(im*4*pi/3) * T9 * gs_power9_rot2) + 0.2*gs_power9_rot2  
    push!(variances9_rot2, variance(T9, gs_power9_rot2))
end
plot!(abs.(variances9_rot2) .+ 1e-16, yaxis=:log)

norm(dot(gs_power9, gs_power9_rot2)) / norm(gs_power9) / norm(gs_power9_rot2)
norm(dot(gs_power9_rot1, gs_power9_rot2)) / norm(gs_power9_rot1) / norm(gs_power9_rot2)

norm.(P9' * gs9)[norm.(P9' * gs9) .> 1e-8]
Λ9[norm.(P9' * gs9) .> 1e-8]
Λ9[(real.(Λ9) .> 1e-8) .* (abs.(imag.(Λ9)) .< 1e-8)]