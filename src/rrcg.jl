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



function rrcg!(A, b::Vector{T}, x::Vector{T}, d, term_min::Float64, term::Float64; 
             init_p::Float64 = 0.0,
             maxIter::Int64=200,
             precon=copy!,
             data=CGData6(length(b), T)) where {T<:Real}

    w_x = copy(x)
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    precon(data.z, data.r)
    data.p .= data.z
    norm_b = genblas_nrm2(b)
    if(term_min == 0.0)
        dice = rand()
        if (dice < init_p)
            return x, w_x, 0
        end
    end  
    mult = 1 / (1-init_p)

    for iter = 0 : maxIter-1
        if (iter < term_min)
            A(data.Ap, data.p)
            gamma = genblas_dot(data.r, data.z)
            alpha = gamma/genblas_dot(data.p, data.Ap)
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            w_x = copy(x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            precon(data.z, data.r)
            beta = genblas_dot(data.z, data.r)/gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
        else
            weight = 1 / ccdf(d, iter)
            A(data.Ap, data.p)
            gamma = genblas_dot(data.r, data.z)
            alpha = gamma/genblas_dot(data.p, data.Ap)
            # x += alpha*p
            genblas_axpy!(mult * weight * alpha, data.p, w_x)
            genblas_axpy!(alpha, data.p, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            if term == iter - 1
                residual_0 = genblas_nrm2(data.r)
                rel_residual = residual_0 / norm_b
                println("res norm: ", rel_residual)
            end
            if term <= iter
                residual_0 = genblas_nrm2(data.r)
                rel_residual = residual_0 / norm_b
                println("res norm: ", rel_residual)
                return x, w_x, iter+1
            end
            precon(data.z, data.r)
            beta = genblas_dot(data.z, data.r)/gamma
            # p = z + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
        end
    end
    return x, w_x, maxIter
end


export CGData6, rrcg!