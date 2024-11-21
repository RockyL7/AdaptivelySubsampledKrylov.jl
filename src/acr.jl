using Distributions
using Statistics
using Random
using LinearAlgebra


# Data container
struct CGData15{T<:Real}
    r_A::Vector{T}
    r_B::Vector{T}
    z::Vector{T}
    p_A::Vector{T}
    p_B::Vector{T}
    Ap::Vector{T}
    Bp::Vector{T}
    Ar::Vector{T}
    Br::Vector{T}
    CGData15(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n))
end

# Solves for x
function acr!(A, b::Vector{T}, x::Vector{T}; term_min::Int64=0, init_p::Float64 = 0.0, maxIter::Int64=200,
             tol::Float64=1e-6, data=CGData15(length(b), T)) where {T<:Real}
    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return x, x, 0
    end
    A(data.r_A, x)
    genblas_scal!(-one(T), data.r_A)
    genblas_axpy!(one(T), b, data.r_A)
    norm_b = genblas_nrm2(b)
    sum_p = 0.0
    weight = 1
    first = 0
    second = 0
    data.r_B .= data.r_A
    data.p_A .= data.r_A
    data.p_B .= data.r_B
    x_B = copy(x)
    alpha_A = 0

    for iter = 0 : term_min + 1
        if (term_min == iter)
            sum_p += init_p
            # Random.seed!(37 * i)
            dice = rand()
            if (dice < init_p)
                return x, x, iter
            end
        end
        weight = 1 / (1 - sum_p)
        A(data.Ap, data.p_A)
        A(data.Ar, data.r_A)
        gamma = genblas_dot(data.r_A, data.Ar)
        Ap_sq = genblas_dot(data.Ap, data.Ap)
        alpha_A = gamma / Ap_sq
        # x += alpha*p
        genblas_axpy!(alpha_A, data.p_A, x)
        genblas_axpy!(weight * alpha_A, data.p_A, x_B)
        # r -= alpha*Ap
        genblas_axpy!(-alpha_A, data.Ap, data.r_A)
        residual = genblas_nrm2(data.r_A) / norm_b
        if isnan(residual)
            return x, x_B, iter+1
        end
        #println(rel_residual)
        #println(iter)
        # if rel_residual <= tol
        #     return x, iter
        # end
        A(data.Ar, data.r_A)
        beta_A = genblas_dot(data.r_A, data.Ar) / gamma
        # p = r + beta*p
        genblas_scal!(beta_A, data.p_A)
        genblas_axpy!(1.0, data.r_A, data.p_A)
        if (iter == term_min)
            first = alpha_A^2 * Ap_sq
        end
        if (iter == term_min + 1)
            second = alpha_A^2 * Ap_sq
        end
    end

    value_d = sqrt(first)
    count = 1

    for iter = term_min + 2 : maxIter-1
        if(first < second)
            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma = genblas_dot(data.r_A, data.Ar)
            Ap_sq = genblas_dot(data.Ap, data.Ap)
            alpha_A = gamma / Ap_sq

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            weight = 1 / (1 - sum_p)
            genblas_axpy!(weight * alpha_A, data.p_A, x_B)

            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            residual = genblas_nrm2(data.r_A) / norm_b
            if isnan(residual)
                return x, x_B, iter+1
            end

            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma
            # p = r + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)
            second = (second * count + alpha_A^2 * Ap_sq) / (count +1)
            count += 1
        else
            value_n = sqrt(first) - sqrt(second)
            p_p = value_n / value_d * (1 - init_p)
            dice = rand()
            if (dice < (p_p / (1 - sum_p)))
                return x, x_B, iter
            end
            sum_p += p_p
            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma = genblas_dot(data.r_A, data.Ar)
            Ap_sq = genblas_dot(data.Ap, data.Ap)
            alpha_A = gamma / Ap_sq

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            weight = 1 / (1 - sum_p)
            genblas_axpy!(weight * alpha_A, data.p_A, x_B)

            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            residual = genblas_nrm2(data.r_A) / norm_b
            if isnan(residual)
                return x, x_B, iter+1
            end

            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma
            # p = r + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)
            first = second
            second = alpha_A^2 * Ap_sq
            count = 1
        end
    end
    weight = 1 / (1 - sum_p)
    genblas_axpy!(weight * alpha_A, data.p_A, x_B)
    return x, x_B, maxIter
end

# # API
# function cr(A, b::Vector{T};
#             tol::Float64=1e-6, maxIter::Int64=200,
#             precon=copy!,
#             data=CGData2(length(b), T)) where {T<:Real}
#     x = zeros(eltype(b), length(b))
#     x, exit_code, num_iters, res_list = cr!(A, b, x, tol=tol, maxIter=maxIter, data=data)
#     return x, exit_code, num_iters, res_list
# end

export CGData15, acr!