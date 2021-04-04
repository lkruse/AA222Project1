#*******************************************************************************
# PACKAGES
#*******************************************************************************
using LinearAlgebra

#*******************************************************************************
# OPTIMIZATION FUNCTION DEFNITIONS
#*******************************************************************************
abstract type DescentMethod end

mutable struct GradientDescent <: DescentMethod
    α0
    γ
end

gd_init!(gd::GradientDescent, f, ∇f, x, k) = gd

function gd_step!(gd::GradientDescent, f, ∇f, x, k)
    #g = ∇f(x)
    α = gd.α0*gd.γ^(k-1)
    return x - α*∇f(x)
end

# Nesterov Momentum
mutable struct NestorovMomentum <: DescentMethod
    α
    β
    v
end
function nm_init!(M::NestorovMomentum,f,gradf,x)
    M.v = zeros(length(x))
    return M
end
function nm_step!(M::NestorovMomentum, f, gradf, x)
    α, β, v = M.α, M.β, M.v
    v[:] = β*v - α*gradf(x + β*v)
    return x + v
end

# Adadelta
mutable struct Adadelta <: DescentMethod
    ys
    yx
    eps
    s
    u
end
function adadelta_init!(M::Adadelta, f, gradf, x)
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M
end
function adadelta_step!(M::Adadelta, f, gradf, x)
    ys, yx, eps, s, u, g = M.ys, M.yx, M.eps, M.s, M.u, gradf(x)
    s[:] = ys*s + (1-ys)*g.*g
    del_x = - (sqrt.(u) .+ eps) ./ (sqrt.(s) .+ eps) .* g
    u[:] = yx*u + (1-yx)*del_x.*del_x
    return x + del_x
end

# HyperNesterovMomentum
mutable struct HyperNesterovMomentum <: DescentMethod
    alpha0
    mu
    beta
    v
    alpha 
    g_prev
end
function hnm_init!(M::HyperNesterovMomentum,f,gradf,x)
    M.alpha = M.alpha0
    M.v = zeros(length(x))
    M.g_prev = zeros(length(x))
    return M
end
function hnm_step!(M::HyperNesterovMomentum, f, gradf, x)
    a,B,u = M.alpha, M.beta, M.mu
    my_v,my_g,my_g_prev = M.v, gradf(x), M.g_prev
    a = a - u*(dot(my_g, (-my_g_prev-B*my_v)))
    println(my_g)
    my_v[:] = B*my_v + my_g
    M.g_prev, M.alpha = my_g, a
    return x - a*(my_g + B*my_v)
end

#ADAM CRAP
mutable struct Adam <: DescentMethod
    alpha
    gamma_v
    gamma_s
    eps
    k
    v
    s
end
function adam_init!(M::Adam, f, gradf, x)
    M.k = 0
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    return M
end
function adam_step!(M::Adam, f, gradf, x)
    alpha, gamma_v, gamma_s, eps, k = M.alpha, M.gamma_v, M.gamma_s, M.eps, M.k
    s,v,g = M.s, M.v, gradf(x)
    v[:] = gamma_v*v + (1-gamma_v)*g
    s[:] = gamma_s*s + (1-gamma_s)*g.*g
    M.k = k += 1
    v_hat = v ./ (1 - gamma_v^k)
    s_hat = s ./ (1 - gamma_s^k)
    return x - alpha*v_hat ./ (sqrt.(s_hat) .+ eps)
end

function test_plot(x, y, x0)
    x1 = -3.5:0.01:3.5
    x2 = -3.5:0.01:3.5
    plot_f(x1, x2) = begin
        (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
        #(1.0 - x1)^2 + 5 * (x2 - x1^2)^2
    end
    X = repeat(reshape(x1, 1, :), length(x2), 1)
    Y = repeat(x2, 1, length(x1))
    Z = map(plot_f, X, Y)
    #print(minimum(Z))
    h = [minimum(Z), 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, maximum(Z)]
    contour(x1, x2, Z, levels=h, fill=true, color=:viridis, linewidth=2.0)
    #contour(x1, x2, Z, nlevels=10, fill=false, color=:viridis, linewidth=2.0)
    #surface(x1, x2, Z, levels=h, fill=false, color=:viridis, linewidth=2.0)
    #heatmap(x1,x2,Z, color = :viridis)
    plot!(x,y,color=:red)
    scatter!([x0[1]], [x0[2]], color=:black, label = "", markersize = 5)
end

function optimize_simple1(f, g, x_best, n)
    x1_arr = Float64[]; x2_arr = Float64[]; y_arr = Float64[];
    #M = GradientDescent(0.001, 0.995)
    #M = NestorovMomentum(0.0006, 0.5, [0.0, 0.0])
    M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    k = 1
    while count(f,g) < n
        #x_best = gd_step!(M, f, g, x_best, k)
        #x_best = nm_step!(M,f,g,x_best)
        x_best = adam_step!(M,f,g,x_best)
        k = k + 1
        push!(x1_arr, x_best[1])
        push!(x2_arr, x_best[2])
    end
    #y_best = f()
    return x_best
end

function optimize_simple2(f, g, x_best, n)
    #M = GradientDescent(0.01, 0.995)
    #M = NestorovMomentum(0.001, 0.9, [0.0, 0.0])
    M = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    k = 1
    while count(f,g) < n
        #x_best = gd_step!(M, f, g, x_best, k)
        #x_best = nm_step!(M, f, g, x_best)
        x_best = adam_step!(M,f,g,x_best)
        k = k + 1
    end
    return x_best
end

function optimize_simple3(f, g, x_best, n)
    #M = GradientDescent(0.001, 0.995)
    #M = NestorovMomentum(0.0009, 0.9, [0.0, 0.0, 0.0, 0.0])
    M = Adam(0.4, 0.6, 0.999, 1e-8, 0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    k = 1
    while count(f,g) < n
        #x_best = gd_step!(M, f, g, x_best, k)
        #x_best = nm_step!(M, f, g, x_best)
        x_best = adam_step!(M,f,g,x_best)
        k = k + 1
    end
    return x_best
end

function optimize_secret1(f, g, x_best, n, len)
    v = [0.0 for x = 1:len]; s = [0.0 for x = 1:len]
    σ = [0.1 for x = 1:len]
    gd = GradientDescent(0.12, 0.999)
    k = 1
    while count(f,g) < n
        x_best = gd_step!(gd, f, g, x_best, k)
        k = k + 1
    end
    return x_best
end

function optimize_secret2(f, g, x_best, n, len)
    v = [0.0 for x = 1:len]; s = [0.0 for x = 1:len]
    M = Adam(0.2, 0.6, 0.999, 1e-8, 0.0, v, s)
    while count(f,g) < n
        x_best = adam_step!(M,f,g,x_best)
    end
    return x_best
end


function optimize(f, g, x0, n, prob)
    x_best = x0; x_arr = Float64[]; y_arr = Float64[]
    push!(x_arr, x0[1]); push!(y_arr, x0[2])
    if prob == "simple1"
        x_best = optimize_simple1(f, g, x0, n)
        #display(test_plot(x_arr, y_arr, x0))
    elseif prob == "simple2"
        x_best = optimize_simple2(f, g, x0, n)
    elseif prob == "simple3"
        x_best = optimize_simple3(f, g, x0, n)
    else
        len = length(x0)
        if len == 50
            x_best = optimize_secret1(f, g, x0, n, len)
        else
            x_best = optimize_secret2(f, g, x0, n, len)
        end
    end
    
    #return x_best, x_arr, y_arr, x0
    return x_best
end

#my_x = optimize(rosenbrock, rosenbrock_gradient, rosenbrock_init(), 20, "simple1");

