using Distributions
using Statistics
using Random
using LinearAlgebra


# Data container
struct CGData15{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    Az::Vector{T}
    CGData15(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n))
end

# Solves for x
function ascr(A, b::Vector{T}, x::Vector{T}; η::Float64 = 0.0, maxIter::Int64=Int(1e10),
    precon=copy!, data=CGData15(length(b), T)) where {T<:Real}
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    if isnan(residual_0)
        return x, x, 0
    end
    max_deter_iter = floor(Int, η)
    term_min = max_deter_iter + 1
    init_prob = 1 - (η - max_deter_iter)
    sum_prob = 0.0  
    weight = 1
    g_prev = 0.0
    g_curr = 0.0 
    precon(data.z, data.r)
    data.p .= data.z
    x_B = copy(x)

    for iter = 0 : term_min + 1
        A(data.Ap, data.p)
        A(data.Az, data.z)
        gamma = genblas_dot(data.r, data.Az)
        Ap_sq = genblas_dot(data.Ap, data.Ap)
        alpha = gamma / Ap_sq
        g_prev = g_curr
        g_curr = alpha^2 * Ap_sq
        if (term_min == iter)
            if (term_min > 0)
                init_prob = max(0, init_prob * (sqrt(g_prev) - sqrt(g_curr)) / sqrt(g_prev))
            end
            sum_prob += init_prob
            dice = rand()
            if (dice < init_prob)
                return x, x, iter+1
            end
        end
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(data.r) / residual_0
        if isnan(residual)
            return x, x_B, iter+1
        end
        weight = 1 / (1 - sum_prob)
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        genblas_axpy!(weight * alpha, data.p, x_B)
        precon(data.z, data.r)
        A(data.Az, data.z)
        beta = genblas_dot(data.r, data.Az) / gamma
        # p = r + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.z, data.p)
    end

    value_d = sqrt(g_prev)
    count = 1

    for iter = term_min + 2 : maxIter-1
        if (g_prev < g_curr)
            A(data.Ap, data.p)
            A(data.Az, data.z)
            gamma = genblas_dot(data.r, data.Az)
            Ap_sq = genblas_dot(data.Ap, data.Ap)
            alpha = gamma / Ap_sq
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
            if isnan(residual)
                return x, x_B, iter+1
            end
            weight = 1 / (1 - sum_prob)
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            genblas_axpy!(weight * alpha, data.p, x_B)
            precon(data.z, data.r)
            A(data.Az, data.z)
            beta = genblas_dot(data.r, data.Az) / gamma
            # p = r + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
            g_curr = (g_curr * count + alpha^2 * Ap_sq) / (count +1)
            count += 1
        else
            value_n = sqrt(g_prev) - sqrt(g_curr)
            curr_prob = value_n / value_d * (1 - init_prob)
            dice = rand()
            if (dice < (curr_prob / (1 - sum_prob)))
                return x, x_B, iter
            end
            sum_prob += curr_prob
            A(data.Ap, data.p)
            A(data.Az, data.z)
            gamma = genblas_dot(data.r, data.Az)
            Ap_sq = genblas_dot(data.Ap, data.Ap)
            alpha = gamma / Ap_sq
            # r -= alpha*Ap
            genblas_axpy!(-alpha, data.Ap, data.r)
            residual = genblas_nrm2(data.r) / residual_0
            if isnan(residual)
                return x, x_B, iter+1
            end
            weight = 1 / (1 - sum_prob)
            # x += alpha*p
            genblas_axpy!(alpha, data.p, x)
            genblas_axpy!(weight * alpha, data.p, x_B)
            precon(data.z, data.r)
            A(data.Az, data.z)
            beta = genblas_dot(data.r, data.Az) / gamma
            # p = r + beta*p
            genblas_scal!(beta, data.p)
            genblas_axpy!(1.0, data.z, data.p)
            g_prev = g_curr
            g_curr = alpha^2 * Ap_sq
            count = 1
        end
    end
    return x, x_B, maxIter
end

export CGData15, ascr