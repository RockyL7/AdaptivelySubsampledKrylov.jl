import LinearAlgebra
using LinearAlgebra

# Data container
struct CGData2{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    Ar::Vector{T}
    CGData2(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n))
end

# Solves for x
function cr!(A, b::Vector{T}, x::Vector{T};
             tol::Float64=1e-6, maxIter::Int64=200,
             data=CGData2(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return x, 1, 0, []
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    rel_residual = residual_0 / residual_0 
    #println(residual_0)
    #println(genblas_nrm2(b))
    res_list = [rel_residual]
    if rel_residual <= tol
        return x, 2, 0, res_list
    end
    data.p .= data.r
    for iter = 1:maxIter
        A(data.Ap, data.p)
        A(data.Ar, data.r)
        gamma = genblas_dot(data.r, data.Ar)
        alpha = gamma/genblas_dot(data.Ap, data.Ap)
        if alpha == Inf || alpha < 0
            return x, -13, iter, res_list
        end
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        rel_residual = genblas_nrm2(data.r)/residual_0
        #println(rel_residual)
        #println(iter)
        res_list = hcat(res_list, rel_residual)
        if rel_residual <= tol
            return x, 30, iter, res_list
        end
        A(data.Ar, data.r)
        beta = genblas_dot(data.r, data.Ar)/gamma
        # p = r + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.r, data.p)
    end
    return x, -2, maxIter, res_list
end

# API
function cr(A, b::Vector{T};
            tol::Float64=1e-6, maxIter::Int64=200,
            precon=copy!,
            data=CGData2(length(b), T)) where {T<:Real}
    x = zeros(eltype(b), length(b))
    x, exit_code, num_iters, res_list = cr!(A, b, x, tol=tol, maxIter=maxIter, data=data)
    return x, exit_code, num_iters, res_list
end

export CGData2, cr!, cr


