using LinearAlgebra
using Distributions
using Statistics



# Data container
struct CGData6{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    CGData6(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n))
end



function rrcg(A, b::Vector{T}, x::Vector{T}; term_min::Float64 = 0.0, 
              λ::Float64 = 0.05, init_prob::Float64 = 0.0,
              maxIter::Int64=200,
              precon=copy!,
              data=CGData6(length(b), T)) where {T<:Real}

    if(term_min == 0.0)
        dice = rand()
        if (dice < init_prob)
            return x, x, 0
        end
    end
    init_weight = 1 / (1 - init_prob)
    n = length(b)
    para = 1 - 1 / exp(λ)
    dist = truncated(Geometric(para), lower = max(term_min, 0.0), upper = n)
    term = rand(dist)
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    if isnan(residual_0)
        return x, x, 0
    end
    precon(data.z, data.r)
    data.p .= data.z
    x_B = copy(x)
    
    for iter = 0 : maxIter - 1
        if (iter < term_min)
            A(data.Ap, data.p)
            gamma = genblas_dot(data.r, data.z)
            alpha = gamma / genblas_dot(data.p, data.Ap)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
            if isnan(residual)
                return x, x_B, iter+1
            end
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            x_B = copy(x)
            precon(data.z, data.r)
            beta = genblas_dot(data.z, data.r) / gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
        else
            A(data.Ap, data.p)
            gamma = genblas_dot(data.r, data.z)
            alpha = gamma/genblas_dot(data.p, data.Ap)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
            if isnan(residual)
                return x, x_B, iter+1
            end
            weight = 1 / ccdf(dist, iter)
            # x += alpha*p
            genblas_axpy!(init_weight * weight * alpha, data.p, x_B)
            genblas_axpy!(alpha, data.p, x)
            if term <= iter
                return x, x_B, iter+1
            end
            precon(data.z, data.r)
            beta = genblas_dot(data.z, data.r) / gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
        end
    end
    return x, x_B, maxIter
end


export CGData6, rrcg