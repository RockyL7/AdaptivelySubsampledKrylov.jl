# Data container
# import Distributions
# import Statistics
# import LinearAlgebra
# using LinearAlgebra
# using Distributions
# using Statistics

struct CGData{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    CGData(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n))
end

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
function rrcg!(A, b::Vector{T}, x::Vector{T}, d, term_min::Float64, term::Float64; 
             init_p::Float64 = 0.0,
             maxIter::Int64=200,
             precon=copy!,
             data=CGData(length(b), T)) where {T<:Real}
    w_x = copy(x)
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        w_x = copy(x)
        #return 1, 0, -1, x
        return x, w_x, 0
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    #sum_res = 1 - residual_0
    #if term <= (sum_res)
    #    return 2, 0, residual_0, x
    #end
    precon(data.z, data.r)
    data.p .= data.z

    dice = rand()
    if (dice < init_p)
        return x, w_x, 0
    end

    mult = 1 / (1-init_p)

    for iter = 1 : maxIter
        if (iter < term_min)
            A(data.Ap, data.p)
            gamma = genblas_dot(data.r, data.z)
            alpha = gamma/genblas_dot(data.p, data.Ap)
            if alpha == Inf || alpha < 0
                #return -13, iter, residual_0, x
                return x, w_x, iter-1
            end
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            w_x = copy(x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r)/residual_0
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
            if alpha == Inf || alpha < 0
                #return -13, iter, residual_0, x
                return x, w_x, iter-1
            end
            # x += alpha*p
            genblas_axpy!(mult * weight * alpha, data.p, w_x)
            genblas_axpy!(alpha, data.p, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
            if term <= (iter)
                #return 30, iter, residual, x
                return x, w_x, iter
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


function rrcr2!(A, b::Vector{T}, x::Vector{T}, d, term_min::Float64, term::Float64;
            init_p::Float64 = 0.0,
            maxIter::Int64=200, 
            precon=copy!,
        data=CGData2(length(b), T)) where {T<:Real}
        w_x = copy(x)
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        w_x = copy(x)
        #return 1, 0, -1, x
        return x, w_x, 0
    end

    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
#sum_res = 1 - residual_0
#if term <= (sum_res)
#    return 2, 0, residual_0, x
#end
    data.p .= data.r

    dice = rand()
    if (dice < init_p)
        return x, w_x, 0
    end

    mult = 1 / (1-init_p)

    for iter = 1:maxIter
        if (iter < term_min)
            A(data.Ap, data.p)
            A(data.Ar, data.r)
            gamma = genblas_dot(data.r, data.Ar)
            alpha = gamma/genblas_dot(data.Ap, data.Ap)
            if alpha == Inf || alpha < 0
                return x, w_x, iter-1
            end
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            w_x = copy(x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r)/residual_0
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
            if alpha == Inf || alpha < 0
                return x, w_x, iter-1
            end
            # x += alpha*p
            genblas_axpy!(mult * weight * alpha, data.p, w_x)
            genblas_axpy!(alpha, data.p, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
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

export CGData, rrcg!, CGData2, rrcr2!