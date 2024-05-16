using Distributions
using Statistics
using Random
using LinearAlgebra


# Data container
struct CGData5{T<:Real}
    r_A::Vector{T}
    z::Vector{T}
    p_A::Vector{T}
    Ap::Vector{T}
    r_B::Vector{T}
    p_B::Vector{T}
    Bp::Vector{T}
    CGData5(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
end




function acg!(A, b::Vector{T}, x::Vector{T};  term_min::Int64=0, init_p::Float64 = 0.0, 
    maxIter::Int64=200, tol::Float64=1e-6, precon=copy!,
    data=CGData5(length(b), T)) where {T<:Real}

    A(data.r_A, x)
    genblas_scal!(-one(T), data.r_A)
    genblas_axpy!(one(T), b, data.r_A)
    p_Anorm_list = []
    P_list = []
    sum_p = 0.0  
    weight = 1
    first = 0
    second = 0
    data.r_B .= data.r_A   
    precon(data.z, data.r_A)
    data.p_A .= data.z
    data.p_B .= data.z
    x_B = copy(x)
    update_p = zeros(size(b))

    for iter = 0 : term_min + 1
        if (term_min == iter)
            sum_p += init_p
            # Random.seed!(37 * i)
            dice = rand()
            if (dice < init_p)
                return x, x, iter
            end
        end
        A(data.Ap, data.p_A)
        gamma_A = genblas_dot(data.r_A, data.z)
        pAp = genblas_dot(data.p_A, data.Ap)
        alpha_A = gamma_A / pAp
        push!(p_Anorm_list, alpha_A^2 * pAp)
        # x += alpha*p
        genblas_axpy!(alpha_A, data.p_A, x)

        if (iter < term_min + 1)
            genblas_axpy!(alpha_A, data.p_A, x_B)
        else
            update_p .= update_p + alpha_A * data.p_A
        end
        # r -= alpha*Ap
        genblas_axpy!(-alpha_A, data.Ap, data.r_A)
        precon(data.z, data.r_A)
        beta_A = genblas_dot(data.z, data.r_A) / gamma_A
        # p = z + beta*p
        genblas_scal!(beta_A, data.p_A)
        genblas_axpy!(1.0, data.z, data.p_A)
        if (iter == term_min)
            first = alpha_A^2 * pAp
        end
        if (iter == term_min + 1)
            second = alpha_A^2 * pAp
        end
    end
    
    value_d = sqrt(first)
    count = 1

    for iter = term_min + 2 : maxIter - 1
        if (first < second)
            A(data.Ap, data.p_A)
            gamma_A = genblas_dot(data.r_A, data.z)
            pAp = genblas_dot(data.p_A, data.Ap)
            alpha_A = gamma_A / pAp
            update_p .= update_p + alpha_A * data.p_A
            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            precon(data.z, data.r_A)
            beta_A = genblas_dot(data.z, data.r_A) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.z, data.p_A)
            second = (second * count + alpha_A^2 * pAp) / (count + 1)
            count += 1
        else 
            ## calculate the weight
            weight = 1 / (1-sum_p)
            genblas_axpy!(weight, update_p, x_B)


            ## roll the dice to decide whether terminate or not
            value_n = sqrt(first) - sqrt(second)
            p_p = value_n / value_d * (1 - init_p)
            dice = rand()
            if (dice < (p_p / (1 - sum_p)))
                return x, x_B, iter
            end
            sum_p += p_p
            A(data.Ap, data.p_A)
            gamma_A = genblas_dot(data.r_A, data.z)
            pAp = genblas_dot(data.p_A, data.Ap)
            alpha_A = gamma_A / pAp
            update_p .= alpha_A * data.p_A
            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            precon(data.z, data.r_A)
            beta_A = genblas_dot(data.r_A, data.z) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.z, data.p_A)
            first = second
            second = alpha_A^2 * pAp
            count = 1
        end
    end
    weight = 1 / (1 - sum_p)
    genblas_axpy!(weight, update_p, x_B)
    return x, x_B, maxIter
end

export CGData5, acg!