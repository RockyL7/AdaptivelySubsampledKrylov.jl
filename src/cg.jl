import LinearAlgebra: BLAS, BlasFloat, norm
using LinearAlgebra

# Data container
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
function cg!(A, b::Vector{T}, x::Vector{T};
             tol::Float64=1e-6, maxIter::Int64=200,
             precon=copy!,
             data=CGData(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return 1, 0, [], []
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    res_list = [residual_0]
    Anorm_list = []
    if residual_0 <= tol
        return 2, 0, res_list, Anorm_list
    end
    
    precon(data.z, data.r)
    data.p .= data.z
    for iter = 1:maxIter
        A(data.Ap, data.p)
        gamma = genblas_dot(data.r, data.z)
        alpha = gamma/genblas_dot(data.p, data.Ap)
        println(alpha)
        if alpha == Inf || alpha < 0
            return -13, iter, res_list, Anorm_list
        end
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(data.r)/residual_0
        res_list = hcat(res_list, residual)
        push!(Anorm_list, alpha^2 * genblas_dot(data.p, data.Ap))
        if residual <= tol
            return 30, iter, res_list, Anorm_list
        end
        #precon(data.z, data.r)
        beta = genblas_dot(data.z, data.r)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.z, data.p)
    end
    return -2, maxIter, res_list, Anorm_list
end

# API
function cg(A, b::Vector{T};
            tol::Float64=1e-6, maxIter::Int64=200,
            precon=copy!,
            data=CGData(length(b), T)) where {T<:Real}
    x = zeros(eltype(b), length(b))
    exit_code, num_iters, res_list = cg!(A, b, x, tol=tol, maxIter=maxIter, precon=precon, data=data)
    return x, exit_code, num_iters, res_list
end

export CGData, cg!, cg


