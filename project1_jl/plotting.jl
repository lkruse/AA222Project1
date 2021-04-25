using LinearAlgebra
using Plots
using Printf

include("helpers.jl")
include("simple.jl")

# Create an abstract type for first-order descent methods
abstract type FirstOrderMethod end

#*******************************************************************************
# Adam Implementation
mutable struct Adam <: FirstOrderMethod
    α; γv; γs; ϵ; k; v; s
end

adam_init!(A::Adam) = A

function adam_step!(A::Adam, ∇f, x)
    α, γv, γs, ϵ, k, s, v, g = A.α , A.γv, A.γs, A.ϵ, A.k, A.s, A.v, ∇f(x)
    v[:] = γv*v + (1-γv)*g
    s[:] = γs*s + (1-γs)*g.*g
    A.k = k += 1
    v̂ = v ./ (1 - γv^k)
    ŝ = s ./ (1 - γs^k)
    return x - α*v̂ ./ (sqrt.(ŝ) .+ ϵ)
end

function optimize_and_store_path(M, f, g, x_best, n)
    x1_arr = Float64[]; x2_arr = Float64[];
    push!(x1_arr, x_best[1]); push!(x2_arr, x_best[2]);
    k = 1
    while k < n
        x_best = adam_step!(M,g,x_best)
        push!(x1_arr, x_best[1]);push!(x2_arr, x_best[2])
        k = k + 1
    end
    return x_best, x1_arr, x2_arr
end

function optimize_and_store_y_vals(M, f, g, x_best, n)
    y_arr = Float64[];
    push!(y_arr, f(x_best))
    k = 1
    while k < n
        x_best = adam_step!(M,g,x_best)
        push!(y_arr, f(x_best))
        k = k + 1
    end
    return y_arr
end

function plot_rosenbrock_path(f, g, n)
    x01 = [3.0, -3.0]; x02 = [0.0, 3.0]; x03 = [-1.7, 2.8]
    M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    x1_best, x11_arr, x21_arr = optimize_and_store_path(M, f, g, x01, n);
    M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    x2_best, x12_arr, x22_arr = optimize_and_store_path(M, f, g, x02, n);
    M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    x3_best, x13_arr, x23_arr = optimize_and_store_path(M, f, g, x03, n);

    # Plot Rosenbrock Plot with select contours
    x1 = -4.0:0.01:4.0
    x2 = -4.0:0.01:4.0
    plot_f(x1, x2) = begin
        (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
    end
    X = repeat(reshape(x1, 1, :), length(x2), 1)
    Y = repeat(x2, 1, length(x1))
    Z = map(plot_f, X, Y)
    #h = [minimum(Z), 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, maximum(Z)]
    h = [minimum(Z), 2.5, 10, 25, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 25000, 35000,maximum(Z)]
    #h = [minimum(Z), 2.5, 25, 250, 500, 1000, 2500, 5000,10000,25000,50000,maximum(Z)]
    contour(x1, x2, Z, levels=h, fill=false, c=cgrad(:viridis, rev = true), linewidth=2.0)

    # Plot first path
    plot!(x11_arr,x21_arr,color=:magenta, label = "x0: (3.0, -3.0)",legend = :bottomleft,
          title = "Adam Method to Minimize Rosenbrock's Function", titlefontsize = 12,
          xlabel = "x1", ylabel = "x2", dpi=300, colorbar = false)
    scatter!([x11_arr], [x21_arr], color=:magenta, label = "", markersize = 3, outline = :magenta)
    scatter!([x01[1]], [x01[2]], color=:magenta,  marker = :rect,label = "", markersize = 5)
    scatter!([x1_best[1]], [x1_best[2]], color=:magenta, marker = :star4, label = "", markersize = 7)

    # Plot second path
    plot!(x12_arr,x22_arr,color=:red, label = "x0: (0.0, 3.0)")
    scatter!([x12_arr], [x22_arr], color=:red, label = "", markersize = 3, outline = false)
    scatter!([x02[1]], [x02[2]], color=:red, marker = :rect, label = "", markersize = 5)
    scatter!([x2_best[1]], [x2_best[2]], color=:red, marker = :star4, label = "", markersize = 7)

    # Plot third path
    plot!(x13_arr,x23_arr,color=:darkorange, label = "x0: (-1.7, 2.8)")
    scatter!([x13_arr], [x23_arr], color=:darkorange, label = "", markersize = 3, outline = false)
    scatter!([x03[1]], [x03[2]], color=:darkorange,  marker = :rect, label = "", markersize = 5)
    scatter!([x3_best[1]], [x3_best[2]], color=:darkorange, marker = :star4, label = "", markersize = 7)

    scatter!([1], [1], color=:dodgerblue, marker = :star5, label = "", markersize = 10)

    png("adam.png")

end

function return_y_vals(f, g, x0, n, prob)
    x_best = x0; x_arr = Float64[]; y_arr = Float64[]
    push!(x_arr, x0[1]); push!(y_arr, x0[2])
    if prob == "simple1"
        M = Adam(0.6, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
        y_arr = optimize_and_store_y_vals(M, f, g, x0, n)
    elseif prob == "simple2"
        M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
        y_arr = optimize_and_store_y_vals(M, f, g, x0, n)
    elseif prob == "simple3"
        M = Adam(0.4, 0.6, 0.999, 1e-8, 0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        y_arr = optimize_and_store_y_vals(M, f, g, x0, n)
    else
        print("Input a simple problem")
        return
    end

    return y_arr
end

n = 100
my_x = plot_rosenbrock_path(rosenbrock, rosenbrock_gradient, n)

x0_r = rosenbrock_init()
x0_r_x1 = round(x0_r[1], digits=3); x0_r_x2 = round(x0_r[2], digits=3)
rosenbrock_convergence = 
    return_y_vals(rosenbrock, rosenbrock_gradient, x0_r, n, "simple1");

x0_h = himmelblau_init()
x0_h_x1 = round(x0_h[1], digits=3); x0_h_x2 = round(x0_h[2], digits=3)
himmelblau_convergence = 
    return_y_vals(himmelblau, himmelblau_gradient, x0_h, n, "simple2");

x0_p = powell_init()
x0_p_x1 = round(x0_p[1], digits=3); x0_p_x2 = round(x0_p[2], digits=3)
x0_p_x3 = round(x0_p[3], digits=3); x0_p_x4 = round(x0_p[4], digits=3)
powell_convergence = 
    return_y_vals(powell, powell_gradient, x0_p, n, "simple3");


rosenbrock_convergence1 = 
    return_y_vals(rosenbrock, rosenbrock_gradient, [3.0, -3.0], n, "simple1");
rosenbrock_convergence2 = 
    return_y_vals(rosenbrock, rosenbrock_gradient, [0.0, 3.0], n, "simple1");
rosenbrock_convergence3 = 
    return_y_vals(rosenbrock, rosenbrock_gradient, [-1.7, 2.8], n, "simple1");
    
himmelblau_convergence1 = 
    return_y_vals(himmelblau, himmelblau_gradient, [3.0, -3.0], n, "simple2");
himmelblau_convergence2 = 
    return_y_vals(himmelblau, himmelblau_gradient, [0.0, 3.0], n, "simple2");
himmelblau_convergence3 = 
    return_y_vals(himmelblau, himmelblau_gradient, [-1.7, 2.8], n, "simple2");

powell_convergence1 = 
    return_y_vals(powell, powell_gradient, [3.0, -3.0, 3.0, -3.0], n, "simple3");
powell_convergence2 = 
    return_y_vals(powell, powell_gradient, [0.0, 3.0, 0.0, 3.0], n, "simple3");
powell_convergence3 = 
    return_y_vals(powell, powell_gradient, [-1.7, 2.8, -1.7, 2.8], n, "simple3");

x_vals = 1:n

plot(x_vals, rosenbrock_convergence1, lw = 3, label = "x0: (3.0, -3.0)",legend = :topright,
    title="Convergence for Rosenbrock Function",xlabel="Iteration",ylabel="f(x)",dpi=300)
plot!(x_vals, rosenbrock_convergence2, lw = 3, label = "x0: (0.0, 3.0)")
plot!(x_vals, rosenbrock_convergence3, lw = 3, label = "x0: (-1.7, 2.8)")
png("rosenbrock_convergence.png")

plot(x_vals, himmelblau_convergence1, lw = 3, label = "x0: (3.0, -3.0)",legend = :topright,
    title="Convergence for Himmelblau Function",xlabel="Iteration",ylabel="f(x)",dpi=300)
plot!(x_vals, himmelblau_convergence2, lw = 3, label = "x0: (0.0, 3.0)")
plot!(x_vals, himmelblau_convergence3, lw = 3, label = "x0: (-1.7, 2.8)")
png("himmelblau_convergence.png")

plot(x_vals, powell_convergence1, lw = 3, label = "x0: (3.0, -3.0, 3.0, -3.0)",legend = :topright,
    title="Convergence for Powell Function",xlabel="Iteration",ylabel="f(x)",dpi=300)
plot!(x_vals, powell_convergence2, lw = 3, label = "x0: (0.0, 3.0, 0.0, 3.0)")
plot!(x_vals, powell_convergence3, lw = 3, label = "x0: (-1.7, 2.8, -1.7, 2.8)")
png("powell_convergence.png")


##

print("HELLO")