import Distributions
import Statistics
import Random
using Distributions
using Statistics
using Random
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

function resrcr_new!(A, b::Vector{T}, x::Vector{T}, v_c::Float64; k::Float64=0.5,
    r_max::Float64=1e-2, seed::Int64 = 1, maxIter::Int64=200, tol::Float64=1e-6,
    data=CGData2(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return 1, 0, -1, x
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    rel_residual = residual_0 / residual_0 
    #println(residual_0)
    #println(genblas_nrm2(b))
    res_list = [rel_residual]

    P_list = []
    



    if rel_residual <= tol
        return 2, 0, residual_0, res_list, x, P_list
    end
    data.p .= data.r


    
    ind = 0
    for iter = 1: maxIter
        # println(weight)
        A(data.Ap, data.p)
        A(data.Ar, data.r)
        gamma = genblas_dot(data.r, data.Ar)
        alpha = gamma/genblas_dot(data.Ap, data.Ap)
        if alpha == Inf || alpha < 0
            return -13, iter, residual_0, res_list, x, P_list
        end
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        rel_residual = genblas_nrm2(data.r)/residual_0
        res_list = hcat(res_list, rel_residual)

        #if rel_residual <= tol
        #    return 30, iter, rel_residual, res_list, x, P_list
        #end
        A(data.Ar, data.r)
        beta = genblas_dot(data.r, data.Ar)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.r, data.p)

        if rel_residual <= r_max
            ind = iter
            break
        end
    end


    p_p = 1
    P_list = [p_p]
    if ind > 1
        for i = 2 : ind
            P_list = hcat(P_list, p_p)
        end
    end

    v_max = v_c

    for iter = ind+1 :maxIter
        v_c = v_max * k * (1-k)^(iter - ind -1)
        a = rel_residual^2
        c = v_c / p_p + ((p_p - 1)^2 / (p_p)^2) * a
        p = -1 / (p_p^2 - 2 * p_p - c * p_p^2 / a)
        #println(v_c)
        #println(p)
        #P_list[J_min + i] = p
        P_list = hcat(P_list, p)
        p_p = p_p * p

        Random.seed!(7 * iter + ind + 35 * seed)
        d = rand()
        #println(d)
        if (d > p)
            return -3, iter-1, rel_residual, res_list, x, P_list
        end


        # println(weight)
        A(data.Ap, data.p)
        A(data.Ar, data.r)
        gamma = genblas_dot(data.r, data.Ar)
        alpha = gamma/genblas_dot(data.Ap, data.Ap)
        if alpha == Inf || alpha < 0
            return -13, iter, residual_0, x, P_list
        end
        weight = 1 / p_p
        # x += alpha*p
        genblas_axpy!(weight*alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        rel_residual = genblas_nrm2(data.r)/residual_0
        res_list = hcat(res_list, rel_residual)

        #if rel_residual <= tol
        #    return 30, iter, rel_residual, res_list, x, P_list
        #end
        A(data.Ar, data.r)
        beta = genblas_dot(data.r, data.Ar)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.r, data.p)
    end
    return -2, maxIter, rel_residual, res_list, x, P_list
end

export CGData, CGData2, resrcr_new!