import Distributions
import Statistics
import Random
import LinearAlgebra
using Distributions
using Statistics
using Random
using LinearAlgebra



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




function resrcg_theory_squared_c4!(A, b::Vector{T}, x::Vector{T};  deter::Int64=0, init_p::Float64 = 0.0,
    seed::Int64 = 1, maxIter::Int64=200, tol::Float64=1e-6, precon=copy!,
    data=CGData5(length(b), T)) where {T<:Real}

    if genblas_nrm2(b) == 0.0
        x .= 0.0
        #return 1, 0, -1, x
        return x, x, 0
    end
    A(data.r_A, x)
    genblas_scal!(-one(T), data.r_A)
    genblas_axpy!(one(T), b, data.r_A)
    residual_0 = genblas_nrm2(data.r_A)
    norm_b = genblas_nrm2(b)
    rel_residual = residual_0 / norm_b # donot need to normalize
    #println(residual_0)
    #println(genblas_nrm2(b))
    p_Anorm_list = []
    if rel_residual <= tol
        #return 2, 0, residual_0, res_list_B, x, P_list
        return x, x, 0
    end

    if (deter == 0)
        P_list = [init_p]
        #Random.seed!(37 * seed)
        d = rand()
        if (d < init_p)
            return x, x, 0
        end
    else
        P_list = [0.0]
    end
    sum_p = P_list[1]
    



    
    data.r_B .= data.r_A   
    precon(data.z, data.r_A)
    data.p_A .= data.z
    data.p_B .= data.z
    x_B = copy(x)

    update_p = zeros(size(b))



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
        gamma_A = genblas_dot(data.r_A, data.z)
        alpha_A = gamma_A / genblas_dot(data.p_A, data.Ap)
        #if alpha_A == Inf || alpha_A < 0
        #    return -13, iter, residual_0, res_list_B, x_B, P_list
        #end
        #p_Anorm_list = hcat(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))
        push!(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))
        # x += alpha*p
        genblas_axpy!(alpha_A, data.p_A, x)

        if (iter < deter + 1)
            genblas_axpy!(alpha_A, data.p_A, x_B)
        else
            update_p .= update_p + alpha_A * data.p_A
        end
        # r -= alpha*Ap
        genblas_axpy!(-alpha_A, data.Ap, data.r_A)


        #if rel_residual_A <= tol
        #    return 30, 1, rel_residual_A, res_list, x, P_list
        #end
        precon(data.z, data.r_A)
        beta_A = genblas_dot(data.z, data.r_A) / gamma_A
        # p = z + beta*p
        genblas_scal!(beta_A, data.p_A)
        genblas_axpy!(1.0, data.z, data.p_A)
        push!(P_list, 0)
    end
    

    value_d =  sqrt(p_Anorm_list[deter + 1])
    first = value_d^2
    second = p_Anorm_list[deter + 2]
    count = 1

    for iter = deter + 2 : maxIter - 1
        if (first < second)
            A(data.Ap, data.p_A)
            gamma_A = genblas_dot(data.r_A, data.z)
            alpha_A = gamma_A / genblas_dot(data.p_A, data.Ap)
            push!(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))
            # p_Anorm_list = hcat(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))
            #if alpha_A == Inf || alpha_A < 0
            #    return -13, iter, residual_0, res_list_B, x_B, P_list
            #end

            update_p .= update_p + alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            precon(data.z, data.r_A)
            beta_A = genblas_dot(data.z, data.r_A) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.z, data.p_A)
            second = (second * count + p_Anorm_list[iter + 1]) / (count + 1)
            count += 1
        else 
            weight = 1 / (1-sum_p)
            genblas_axpy!(weight, update_p, x_B)

            value_n = sqrt(first) - sqrt(second)
            p_p = value_n / value_d

            if (count > 1)
                for i = 1 : count -1 
                    push!(P_list, 0)
                    # P_list = hcat(P_list, 0)
                end
            end
            push!(P_list, p_p)
            #P_list = hcat(P_list, p_p)
            d = rand()
            if (d < (p_p / (1 - sum_p)))
                return x, x_B, iter
            end

            sum_p += p_p

            A(data.Ap, data.p_A)
            gamma_A = genblas_dot(data.r_A, data.z)
            alpha_A = gamma_A / genblas_dot(data.p_A, data.Ap)
            push!(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))
            #p_Anorm_list = hcat(p_Anorm_list, alpha_A^2 * genblas_dot(data.p_A, data.Ap))

            update_p .= alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            precon(data.z, data.r_A)
            beta_A = genblas_dot(data.r_A, data.z) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.z, data.p_A)

            first = second
            second = p_Anorm_list[iter + 1]
            count = 1
        end
    end
    genblas_axpy!(weight, update_p, x_B)
    return x, x_B, maxIter
end

export CGData5, resrcg_theory_squared_c4!