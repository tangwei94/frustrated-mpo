using LinearAlgebra, TensorKit, MPSKit, MPSKitModels, KrylovKit
using ChainRules, ChainRulesCore, TensorKitAD, Zygote, OptimKit
using JLD2
using CairoMakie
using Makie.GeometryBasics

function plot_delta_tensor!(ax, x, y, a)
    scatter!(ax, [x], [y], color = :black, marker=:circle, markersize = 10)
    lines!(ax, [x, x], [y, y-a], color = :black, linewidth = 1)
    lines!(ax, [x, x+sqrt(3)/2*a], [y, y+a/2], color = :black, linewidth = 1)
    lines!(ax, [x, x-sqrt(3)/2*a], [y, y+a/2], color = :black, linewidth = 1)
end

function plot_utriangle!(ax, x, y, a; oriented = true)
    x1 = x + a / 2 
    y1 = y 

    x2 = x 
    y2 = y + a * sqrt(3) / 2

    x3 = x - a / 2
    y3 = y
    if oriented
        poly!(ax, Point2f[(x1, y1), (x2, y2), (x3, y3)], color = :lightskyblue, strokecolor = :black, strokewidth = 1)
    else 
        poly!(ax, Point2f[(x1, y1), (x2, y2), (x3, y3)], color = :bisque, strokecolor = :black, strokewidth = 1)
    end
end

function plot_dtriangle!(ax, x, y, a; oriented=true, error=false)
    x1 = x + a / 2
    y1 = y 

    x2 = x
    y2 = y - a * sqrt(3) / 2

    x3 = x - a / 2
    y3 = y

    if !error
        if oriented
            poly!(ax, Point2f[(x1, y1), (x2, y2), (x3, y3)], color = :bisque, strokecolor = :black, strokewidth = 1)
        else 
            poly!(ax, Point2f[(x1, y1), (x2, y2), (x3, y3)], color = :lightskyblue, strokecolor = :black, strokewidth = 1)
        end
    else
        poly!(ax, Point2f[(x1, y1), (x2, y2), (x3, y3)], color = :red, strokecolor = :black, strokewidth = 1)
    end
end

c = 0.6

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 343.33))
ax1 = Axis(fig[1, 1])

xlims!(ax1, -2.5, 6.5)
ylims!(ax1, -0.5, 4.65)

base_y = 2.65
text!(ax1, -2.4, base_y+3.9*c*sqrt(3)/2, text=L"\text{(a)}", align=(:left, :top))
for ix in -3:2
    plot_utriangle!(ax1, c*(ix), base_y + c*0, c)
    plot_dtriangle!(ax1, c*(ix+1/2), base_y + c*sqrt(3)/2, c)
    plot_utriangle!(ax1, c*(ix+1/2), base_y + c*sqrt(3)/2, c)
    plot_dtriangle!(ax1, c*(ix+1), base_y + c*sqrt(3), c)
    plot_utriangle!(ax1, c*(ix+1), base_y + c*sqrt(3), c)
    plot_dtriangle!(ax1, c*(ix+3/2), base_y + c*3*sqrt(3)/2, c)
end

base_y = 2.6
base_x = 5
text!(ax1, 2.5, base_y+0.05+3.9*c*sqrt(3)/2, text=L"\text{(b)}", align=(:left, :top))
Δy = c*2.5*sqrt(3)/4
a = c*0.6 # length of triangle 
a1 = (Δy - a * sqrt(3)/2) / 1.5 # length of link of delta tensor
for ix in -3:0
    plot_utriangle!(ax1, base_x + c*1.25*ix, base_y + 0, a)
    plot_dtriangle!(ax1, base_x + c*1.25*ix+a/2, base_y + sqrt(3)/2*a, a)
    plot_utriangle!(ax1, base_x + c*1.25*(ix+1/2), base_y + Δy, a)
    plot_dtriangle!(ax1, base_x + c*1.25*(ix+1/2)+a/2, base_y + Δy + sqrt(3)/2*a, a)
    plot_utriangle!(ax1, base_x + c*1.25*(ix+1), base_y + 2*Δy, a)
    plot_dtriangle!(ax1, base_x + c*1.25*(ix+1)+a/2, base_y + 2*Δy + sqrt(3)/2*a, a)

    lines!(ax1, base_x .+ [c*1.25*(ix-1/2)+a/2, c*1.25*(ix)], [base_y-Δy+sqrt(3)/2*a, base_y-Δy+Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*ix+a/2, c*1.25*(ix+1/2)], [base_y+sqrt(3)/2*a, base_y+Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*(ix+1/2)+a/2, c*1.25*(ix+1)], [base_y+Δy+sqrt(3)/2*a, base_y+2*Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*(ix+1)+a/2, c*1.25*(ix+3/2)], [base_y+2*Δy+sqrt(3)/2*a, base_y+3*Δy], color = :black, linewidth = 1)

end
for ix in -4:0
    lines!(ax1, base_x .+ [c*1.25*ix+3/4*a, c*1.25*(ix+1)-a/4], [base_y+sqrt(3)/4*a, base_y+sqrt(3)/4*a], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*(ix+1/2)+3/4*a, c*1.25*(ix+3/2)-a/4], [base_y+sqrt(3)/4*a+Δy, base_y+sqrt(3)/4*a+Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*(ix+1)+3/4*a, c*1.25*(ix+2)-a/4], [base_y+sqrt(3)/4*a+2*Δy, base_y+sqrt(3)/4*a+2*Δy], color = :black, linewidth = 1)
end

base_x = 0.5
base_y = -0.05
text!(ax1, -2.4, base_y+3.9*c*sqrt(3)/2, text=L"\text{(c)}", align=(:left, :top))
Δy = c*2.5*sqrt(3)/4
a = c*0.6 # length of triangle 
a1 = (Δy - a * sqrt(3)/2) / 1.5 # length of link of delta tensor
for ix in -3:0
    plot_utriangle!(ax1, base_x + 1.25*c*ix, base_y + 0, a)
    plot_utriangle!(ax1, base_x + 1.25*c*(ix+1/2), base_y + Δy, a)
    plot_utriangle!(ax1, base_x + 1.25*c*(ix+1), base_y + 2*Δy, a)
    plot_delta_tensor!(ax1, base_x + 1.25*c*(ix-1/2), base_y - Δy + a * sqrt(3)/2 + a1, a1)
    plot_delta_tensor!(ax1, base_x + 1.25*c*ix, base_y + a * sqrt(3)/2 + a1, a1)
    plot_delta_tensor!(ax1, base_x + 1.25*c*(ix+1/2), base_y + Δy + a * sqrt(3)/2 + a1, a1)

    lines!(ax1, base_x .+ [1.25*c*(ix+1), 1.25*c*(ix+1)], [base_y+2*Δy+sqrt(3)/2*a, base_y+2*Δy+a1+sqrt(3)/2*a], color = :black, linewidth = 1)
end


base_y = 1.15
base_x = 5.7
text!(ax1, 2., base_y-1.2+3.9*c*sqrt(3)/2, text=L"\text{(d)}", align=(:left, :top))
Δy = c*2.5*sqrt(3)/4
a = c*0.6 # length of triangle 
a1 = (Δy - a * sqrt(3)/2) / 1.5 # length of link of delta tensor
for ix in -3:0
    plot_dtriangle!(ax1, base_x + c*1.25*ix, base_y + sqrt(3)/2*a - 2*Δy, a)
    plot_utriangle!(ax1, base_x + c*1.25*ix+a/2, base_y - 2*Δy, a)

    lines!(ax1, base_x .+ [c*1.25*(ix-1/2)+a, c*1.25*(ix)+a/2], -2*Δy .+ [base_y-Δy+sqrt(3)/2*a, base_y-Δy+Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*ix, c*1.25*(ix+1/2)-a/2], -2*Δy .+ [base_y+sqrt(3)/2*a, base_y+Δy], color = :black, linewidth = 1)
    
    plot_utriangle!(ax1, base_x + c*1.25*ix, base_y + 0, a)
    plot_dtriangle!(ax1, base_x + c*1.25*ix+a/2, base_y + sqrt(3)/2*a, a)

    lines!(ax1, base_x .+ [c*1.25*(ix-1/2)+a/2, c*1.25*(ix)], [base_y-Δy+sqrt(3)/2*a, base_y-Δy+Δy], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*ix+a/2, c*1.25*(ix+1/2)], [base_y+sqrt(3)/2*a, base_y+Δy], color = :black, linewidth = 1)
    
end
for ix in -4:0
    lines!(ax1, base_x .+ [c*1.25*ix+3/4*a, c*1.25*(ix+1)-a/4], -2*Δy .+ [base_y+sqrt(3)/4*a, base_y+sqrt(3)/4*a], color = :black, linewidth = 1)
    lines!(ax1, base_x .+ [c*1.25*ix+3/4*a, c*1.25*(ix+1)-a/4], [base_y+sqrt(3)/4*a, base_y+sqrt(3)/4*a], color = :black, linewidth = 1)
end
text!(ax1, base_x - 1.25*4*c+3/4*a-a/4, base_y + sqrt(3)/4*a+0.1*a, text=L"\mathcal{T}_1=", align=(:right, :center))
text!(ax1, base_x - 1.25*4*c+3/4*a-a/4, base_y + sqrt(3)/4*a+0.1*a -2*Δy, text=L"\mathcal{T}^\dagger_1=", align=(:right, :center))

hidespines!(ax1)
hidedecorations!(ax1)

save("gauge_AF_triangular_ising/data/fig-tensor-construction.pdf", fig)
@show fig

function plot_spin!(ax, x, y, a; orientation=:up)
    b = a / 2
    if orientation == :up 
        lines!(ax, [x, x], [y, y+a], color=:royalblue)
        lines!(ax, [x, x+b*sin(π/10)], [y+a, y+a-cos(π/10)*b], color=:royalblue)
        lines!(ax, [x, x-b*sin(π/10)], [y+a, y+a-cos(π/10)*b], color=:royalblue)
    elseif orientation == :down
        lines!(ax, [x, x], [y, y+a], color=:darkorange)
        lines!(ax, [x, x+b*sin(π/10)], [y, y+cos(π/10)*b], color=:darkorange)
        lines!(ax, [x, x-b*sin(π/10)], [y, y+cos(π/10)*b], color=:darkorange)
    end
end
function plot_doublespin!(ax, x, y, a)
    plot_spin!(ax, x-a/4, y, a; orientation=:up)
    plot_spin!(ax, x+a/4, y, a; orientation=:down)
end

c = 0.62

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600))
ax1 = Axis(fig[1, 1])

xlims!(ax1, -4.6, 4.4)
ylims!(ax1, -6.8, 2.2)

function plot_transfermat!(ax, base_x, base_y, spins1, spins2; showing_downward_triangles=false, errorat=-999)
    for (ix, s) in zip(-3:2, spins2)
        plot_utriangle!(ax1, base_x + c*(ix), base_y + c*0, c)
        if showing_downward_triangles && ix != 2
            if ix != errorat
                plot_dtriangle!(ax1, base_x + c*(ix+1/2), base_y + c*sqrt(3)/2, c)
            else 
                plot_dtriangle!(ax1, base_x + c*(ix+1/2), base_y + c*sqrt(3)/2, c; error=true)
            end
        end
        if s == 1
            plot_spin!(ax1, base_x + c*(ix), base_y+c*sqrt(3)/2 + c/6, c/3; orientation=:up)
        elseif s == -1
            plot_spin!(ax1, base_x + c*(ix), base_y+c*sqrt(3)/2 + c/6, c/3; orientation=:down)
        elseif s == 0
            plot_doublespin!(ax1, base_x + c*(ix), base_y+c*sqrt(3)/2 + c/6, c/3)
        end
    end
    for (ix, s) in zip(-3:3, spins1)
        if s == 1
            plot_spin!(ax1, base_x + c*(ix)-c/2, base_y-c/2, c/3)
        elseif s == -1
            plot_spin!(ax1, base_x + c*(ix)-c/2, base_y-c/2, c/3; orientation=:down)
        elseif s == 0
            plot_doublespin!(ax1, base_x + c*(ix)-c/2, base_y-c/2, c/3)
        end
    end
end

plot_transfermat!(ax1, -2, 2*c, 
    [1, 1, 1, 1, 1, 1, 1], 
    [-1, -1, -1, -1, -1, -1])
plot_transfermat!(ax1, 2.5, 2*c, 
    [1, 1, 1, 1, 1, 1, 1], 
    [-1, -1, -1, -1, -1, -1]; 
    showing_downward_triangles=true)
plot_transfermat!(ax1, -2, -0.5*c, 
    [1, 1, 1, -1, -1, 1, 1], 
    [-1, -1, 0, 1, 0, -1])
plot_transfermat!(ax1, 2.5, -0.5*c, 
    [1, 1, 1, -1, -1, 1, 1], 
    [-1, -1, 0, 1, 0, -1]; 
    showing_downward_triangles=true)

plot_transfermat!(ax1, -2, -3*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, 1, 1, -1, -1])
plot_transfermat!(ax1, -2, -5.5*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, 1, -1, -1, -1])
plot_transfermat!(ax1, -2, -8*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, -1, 1, -1, -1])
plot_transfermat!(ax1, -2, -10.5*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, -1, -1, -1, -1])
plot_transfermat!(ax1, 2.5, -3*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, 1, 1, -1, -1]; 
    showing_downward_triangles=true)
plot_transfermat!(ax1, 2.5, -5.5*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, 1, -1, -1, -1]; 
    showing_downward_triangles=true)
plot_transfermat!(ax1, 2.5, -8*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, -1, 1, -1, -1]; 
    showing_downward_triangles=true)
plot_transfermat!(ax1, 2.5, -10.5*c, 
    [1, 1, 1, -1, 1, 1, 1], 
    [-1, -1, -1, -1, -1, -1]; 
    showing_downward_triangles=true, errorat=-1)

for (loc, text) in zip(3.5:-2.5:-10.5, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])
    text!(ax1, -4.49, loc*c, text=latexstring("\\text{$(text)}"), align=(:left, :top))
end

hidespines!(ax1)
hidedecorations!(ax1)

save("gauge_AF_triangular_ising/data/fig-transfer-matrices.pdf", fig)
@show fig