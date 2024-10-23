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


function cg!(A, b::Vector{T}, x::Vector{T};
             tol::Float64=1e-6, maxIter::Int64=200,
             precon=copy!,
             data=CGData(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return x, 0, res_list, x_list
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    norm_b = genblas_nrm2(b)
    rel_residual = residual_0 / norm_b
    rel_list = []
    x_list = []
    if rel_residual <= tol
        return x, 0 
    end
    push!(rel_list, rel_residual)
    push!(x_list, x)

    
    precon(data.z, data.r)
    data.p .= data.z
    for iter = 1 : maxIter
        A(data.Ap, data.p)
        gamma = genblas_dot(data.r, data.z)
        alpha = gamma/genblas_dot(data.p, data.Ap)
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(data.r) / norm_b
        push!(rel_list, rel_residual)
        push!(x_list, x)
        if residual <= tol
            return x, iter, rel_list, x_list
        end
        precon(data.z, data.r)
        beta = genblas_dot(data.z, data.r)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.z, data.p)
    end
    return x, maxIter, rel_list, x_list
end


# function cg(A, b::Vector{T};
#             tol::Float64=1e-6, maxIter::Int64=200,
#             precon=copy!,
#             data=CGData(length(b), T)) where {T<:Real}
#     x = zeros(eltype(b), length(b))
#     exit_code, x, num_iters, = cg!(A, b, x; tol=tol, maxIter=maxIter, precon=precon, data=data)
#     return exit_code, x, num_iters 
# end

export CGData, cg!


