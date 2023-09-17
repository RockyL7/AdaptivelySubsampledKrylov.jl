import Distributions
import Statistics
import Random
import LinearAlgebra
using LinearAlgebra
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

struct CGData3{T<:Real}
    r_A::Vector{T}
    p_A::Vector{T}
    Ap::Vector{T}
    Ar::Vector{T}
    r_B::Vector{T}
    p_B::Vector{T}
    Bp::Vector{T}
    Br::Vector{T}
    CGData3(n::Int, T::Type) = new{T}(
        zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
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



function resrcr_theory_squared_c!(A, b::Vector{T}, x::Vector{T};  deter::Int64=0,
    seed::Int64 = 1, maxIter::Int64=200, tol::Float64=1e-6,
    data=CGData3(length(b), T)) where {T<:Real}

    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return 1, 0, -1, x
    end
    A(data.r_A, x)
    genblas_scal!(-one(T), data.r_A)
    genblas_axpy!(one(T), b, data.r_A)
    residual_0 = genblas_nrm2(data.r_A)
    norm_b = genblas_nrm2(b)
    rel_residual = residual_0 / norm_b
    #println(residual_0)
    #println(genblas_nrm2(b))
    res_list_A = [rel_residual]


    sum_p = 0
    P_list = [0]
    



    if rel_residual <= tol
        return 2, 0, residual_0, res_list_B, x, P_list
    end
    data.r_B .= data.r_A
    data.p_A .= data.r_A
    data.p_B .= data.r_A
    x_B = copy(x)

    update_p = zeros(size(b))
    update_q = zeros(size(b))



    # for iter = 0 : deter

    #     B(data.Bp, data.p_B)
    #     B(data.Br, data.r_B)
    #     gamma_B = genblas_dot(data.r_B, data.Br)
    #     alpha_B = gamma_B/genblas_dot(data.Bp, data.Bp)
    #     #if alpha_A == Inf || alpha_A < 0
    #     #    return -13, iter, residual_0, res_list_B, x_b, P_list
    #     #end

    #     # x += alpha*p
    #     genblas_axpy!(alpha_B, data.p_B, x_B)
    #     # r -= alpha*Ap
    #     genblas_axpy!(-alpha_B, data.Bp, data.r_B)
    #     #rel_residual_B = genblas_nrm2(data.r_B) / norm_b
    #     #res_list_B = hcat(res_list_B, rel_residual_B)

    #     #if rel_residual_A <= tol
    #     #    return 30, 1, rel_residual_A, res_list, x, P_list
    #     #end
    #     B(data.Br, data.r_B)
    #     beta_B = genblas_dot(data.r_B, data.Br) / gamma_B
    #     # p = z + beta*p
    #     genblas_scal!(beta_B, data.p_B)
    #     genblas_axpy!(1.0, data.r_B, data.p_B)
    # end




    for iter = 0 : deter + 1

        A(data.Ap, data.p_A)
        A(data.Ar, data.r_A)
        gamma_A = genblas_dot(data.r_A, data.Ar)
        alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
        #if alpha_A == Inf || alpha_A < 0
        #    return -13, iter, residual_0, res_list_B, x_B, P_list
        #end

        if (iter == deter + 1)
            update_p .= alpha_A * data.p_A
        end

        # x += alpha*p
        genblas_axpy!(alpha_A, data.p_A, x)

        if (iter != deter + 1)
            genblas_axpy!(alpha_A, data.p_A, x_B)
        end
        # r -= alpha*Ap
        genblas_axpy!(-alpha_A, data.Ap, data.r_A)
        rel_residual_A = genblas_nrm2(data.r_A) / norm_b
        res_list_A = hcat(res_list_A, rel_residual_A)

        #if rel_residual_A <= tol
        #    return 30, 1, rel_residual_A, res_list, x, P_list
        #end
        A(data.Ar, data.r_A)
        beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
        # p = z + beta*p
        genblas_scal!(beta_A, data.p_A)
        genblas_axpy!(1.0, data.r_A, data.p_A)
    end

    value_d = sqrt(res_list_A[deter+1]^2 - res_list_A[deter + 2]^2)
    curr = 0
    curr = value_d^2
    prev = 0
    pointer = 1


    for iter = deter + 1 : maxIter
        if (pointer == 1)
            prev = curr
            curr = res_list_A[iter + 1]^2 - res_list_A[iter + 2]^2
        else
            curr *= (pointer-1)
            curr += res_list_A[iter + 1]^2 - res_list_A[iter + 2]^2
            curr /= pointer
        end
        
        if (prev < curr)
            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma_A = genblas_dot(data.r_A, data.Ar)
            alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
            #if alpha_A == Inf || alpha_A < 0
            #    return -13, iter, residual_0, res_list_B, x_B, P_list
            #end

            update_q .= update_q + alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            rel_residual_A = genblas_nrm2(data.r_A) / norm_b
            res_list_A = hcat(res_list_A, rel_residual_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)
            pointer = pointer + 1
        else
            value_n = sqrt(prev) - sqrt(curr)
            p_p = value_n / value_d
            P_list = hcat(P_list, p_p)
            Random.seed!(7 * (iter + 2 - pointer) + 1 + 35 * seed)
            d = rand()

            if (d < (p_p / (1 - sum_p)))
                return -13, iter + 1 - pointer, res_list_A, x_B, P_list
            end

            sum_p += p_p
            weight = 1 / (1-sum_p)
            genblas_axpy!(weight, update_p + update_q, x_B)

            if (pointer > 1)
                for i = 2 : pointer
                    P_list = hcat(P_list, 0)
                end
            end
            
            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma_A = genblas_dot(data.r_A, data.Ar)
            alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
            #if alpha_A == Inf || alpha_A < 0
            #    return -13, iter, residual_0, res_list_B, x_B, P_list
            #end

            update_p .= alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            rel_residual_A = genblas_nrm2(data.r_A) / norm_b
            res_list_A = hcat(res_list_A, rel_residual_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)

            update_q .= zeros(size(b))
            pointer = 1
        end
    end

    return -2, maxIter, res_list_A, x_B, P_list
end

export CGData, CGData2, CGData3, resrcr_theory_squared_c!