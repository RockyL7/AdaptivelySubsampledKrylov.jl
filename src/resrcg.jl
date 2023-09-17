# Data container
import Distributions
import Statistics
import LinearAlgebra
using LinearAlgebra
using Distributions
using Statistics
struct CGData{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    CGData(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n))
end

# Solves for x
function resrcg!(A, b::Vector{T}, x::Vector{T}, d, term::Float64;
             maxIter::Int64=200,
             precon=copy!,
             data=CGData(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return 1, 0, -1, x
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    sum_res = 1 - residual_0
    if term <= (sum_res)
        return 2, 0, residual_0, x
    end
    precon(data.z, data.r)
    data.p .= data.z

    for iter = 1:maxIter
        weight = 1 / ccdf(d, sum_res)
        # println(weight)
        A(data.Ap, data.p)
        gamma = genblas_dot(data.r, data.z)
        alpha = gamma/genblas_dot(data.p, data.Ap)
        if alpha == Inf || alpha < 0
            return -13, iter, residual_0, x
        end
        # x += alpha*p
        genblas_axpy!(weight*alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(data.r)/residual_0
        sum_res = 1 - residual
        if term <= (sum_res)
            return 30, iter, residual, x
        end
        precon(data.z, data.r)
        beta = genblas_dot(data.z, data.r)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.z, data.p)
    end
    return -2, maxIter, residual, x
end

export CGData, resrcg!