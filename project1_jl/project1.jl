#*******************************************************************************
# PACKAGES
#*******************************************************************************
using LinearAlgebra

#*******************************************************************************
# OPTIMIZATION FUNCTION DEFNITIONS
#*******************************************************************************
# Create an abstract type for first-order descent methods
abstract type FirstOrderMethod end

#*******************************************************************************
# Gradient Descent Implementation
mutable struct GradientDescent <: FirstOrderMethod
    α0; γ
end

gd_init!(GD::GradientDescent) = GD

function gd_step!(GD::GradientDescent, ∇f, x, k)
    α = GD.α0*GD.γ^(k-1)
    return x - α*∇f(x)
end

#*******************************************************************************
# Nesterov Momentum Implementation
mutable struct NesterovMomentum <: FirstOrderMethod
    α; β; v
end

nm_init!(NM::NesterovMomentum) = NM

function nm_step!(NM::NesterovMomentum, ∇f, x)
    α, β, v = NM.α, NM.β, NM.v
    v[:] = β*v - α*∇f(x + β*v)
    return x + v
end

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

#*******************************************************************************
# Function to optimize the first simple problem
function optimize_simple1(f, g, x_best, n)
    #GD = GradientDescent(0.001, 0.995)
    #NM = NesterovMomentum(0.0006, 0.7, [0.0, 0.0])
    A = Adam(0.6, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    k = 1
    while count(f, g) < n
        #x_best = gd_step!(GD, g, x_best, k)
        #x_best = nm_step!(NM, g, x_best)
        x_best = adam_step!(A, g, x_best)
        k = k + 1
    end
    return x_best
end

#*******************************************************************************
# Function to optimize the second simple problem
function optimize_simple2(f, g, x_best, n)
    #GD = GradientDescent(0.01, 0.995)
    #NM = NesterovMomentum(0.001, 0.9, [0.0, 0.0])
    A = Adam(0.5, 0.4, 0.999, 1e-8, 0.0, [0.0,0.0], [0.0,0.0])
    k = 1
    while count(f, g) < n
        #x_best = gd_step!(GD, g, x_best, k)
        #x_best = nm_step!(NM, g, x_best)
        x_best = adam_step!(A, g, x_best)
        k = k + 1
    end
    return x_best
end

#*******************************************************************************
# Function to optimize the third simple problem
function optimize_simple3(f, g, x_best, n)
    #GD = GradientDescent(0.001, 0.995)
    NM = NesterovMomentum(0.0009, 0.9, [0.0, 0.0, 0.0, 0.0])
    #A = Adam(0.4, 0.6, 0.999, 1e-8, 0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    k = 1
    while count(f, g) < n
        #x_best = gd_step!(GD, g, x_best, k)
        x_best = nm_step!(NM, g, x_best)
        #x_best = adam_step!(A, g, x_best)
        k = k + 1
    end
    return x_best
end

#*******************************************************************************
# Function to optimize the first secret problem
function optimize_secret1(f, g, x_best, n, len)
    v = [0.0 for x = 1:len]; s = [0.0 for x = 1:len]
    σ = [0.1 for x = 1:len]
    #GD = GradientDescent(0.12, 0.999)
    A = Adam(0.16, 0.75, 0.9, 1e-8, 0.0, v, s)
    k = 1
    while count(f, g) < n
        #x_best = gd_step!(GD, g, x_best, k)
        x_best = adam_step!(A, g, x_best)
        k = k + 1
    end
    return x_best
end

#*******************************************************************************
# Function to optimize the second secret problem
function optimize_secret2(f, g, x_best, n, len)
    v = [0.0 for x = 1:len]; s = [0.0 for x = 1:len]
    A = Adam(0.2, 0.6, 0.999, 1e-8, 0.0, v, s)
    while count(f, g) < n
        x_best = adam_step!(A, g, x_best)
    end
    return x_best
end

#*******************************************************************************
# MAIN OPTIMIZATION FUNCTION
#*******************************************************************************
function optimize(f, g, x0, n, prob)
    if prob == "simple1"
        x_best = optimize_simple1(f, g, x0, n)
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
    
    return x_best
end
