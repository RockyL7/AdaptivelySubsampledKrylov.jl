# Data container
# import Distributions
# import Statistics
# import LinearAlgebra
using LinearAlgebra
using Distributions
using Statistics


struct CGData4{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    Ar::Vector{T}
    CGData4(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n))
end


function rrcr!(A, b::Vector{T}, x::Vector{T}, d, term_min::Float64, term::Float64;
            init_p::Float64 = 0.0,
            maxIter::Int64=200, 
        data=CGData4(length(b), T)) where {T<:Real}

    w_x = copy(x)
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
#sum_res = 1 - residual_0
#if term <= (sum_res)
#    return 2, 0, residual_0, x
#end
    data.p .= data.r
    if(term_min == 0.0)
        dice = rand()
        if (dice < init_p)
            return x, w_x, 0
        end
    end

    mult = 1 / (1-init_p)

    for iter = 1:maxIter
        if (iter < term_min)
            A(data.Ap, data.p)
            A(data.Ar, data.r)
            gamma = genblas_dot(data.r, data.Ar)
            alpha = gamma/genblas_dot(data.Ap, data.Ap)
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            w_x = copy(x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            A(data.Ar, data.r)
            beta = genblas_dot(data.r, data.Ar)/gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.r, data.p)
        else
            weight = 1 / ccdf(d, iter)
            A(data.Ap, data.p)
            A(data.Ar, data.r)
            gamma = genblas_dot(data.r, data.Ar)
            alpha = gamma/genblas_dot(data.Ap, data.Ap)
            # x += alpha*p
            genblas_axpy!(mult * weight * alpha, data.p, w_x)
            genblas_axpy!(alpha, data.p, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            if term <= (iter)
                return x, w_x, iter
            end
            A(data.Ar, data.r)
            beta = genblas_dot(data.r, data.Ar)/gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.r, data.p)
        end
    end
    return x, w_x, maxIter
end

export CGData4, rrcr!