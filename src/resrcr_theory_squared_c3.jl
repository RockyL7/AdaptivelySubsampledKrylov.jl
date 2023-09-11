import Distributions
import Statistics
import Random
using Distributions
using Statistics
using Random



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
        zeros(T, n), zeros(T, n), zeros(T, n), 
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
end





function resrcr_theory_squared_c3!(A, b::Vector{T}, x::Vector{T};  deter::Int64=0, init_p::Float64 = 0.0,
    seed::Int64 = 1, maxIter::Int64=200, tol::Float64=1e-6,
    data=CGData3(length(b), T)) where {T<:Real}

    if genblas_nrm2(b) == 0.0
        x .= 0.0
        return x, x, 0
    end
    A(data.r_A, x)
    genblas_scal!(-one(T), data.r_A)
    genblas_axpy!(one(T), b, data.r_A)
    residual_0 = genblas_nrm2(data.r_A)
    norm_b = genblas_nrm2(b)
    rel_residual = residual_0 # donot need to normalize
    #println(residual_0)
    #println(genblas_nrm2(b))
    res_list_A = [rel_residual]
    if rel_residual <= tol
        return x, x, 0
    end

    if (deter == -1)
        P_list = [init_p]
        # Random.seed!(37 * i)
        d = rand()
        if (d < init_p)
            return x, x, 0
        end
    else
        P_list = [0]
    end
    sum_p = P_list[1]
    
    data.r_B .= data.r_A
    data.p_A .= data.r_A
    data.p_B .= data.r_A
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




    for iter = 0 : deter + 2

        A(data.Ap, data.p_A)
        A(data.Ar, data.r_A)
        gamma_A = genblas_dot(data.r_A, data.Ar)
        alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
        #if alpha_A == Inf || alpha_A < 0
        #    return -13, iter, residual_0, res_list_B, x_B, P_list
        #end

        if (iter > deter)
            update_p .= update_p + alpha_A * data.p_A
        end

        # x += alpha*p
        genblas_axpy!(alpha_A, data.p_A, x)

        if (iter < deter + 1)
            genblas_axpy!(alpha_A, data.p_A, x_B)
        end
        # r -= alpha*Ap
        genblas_axpy!(-alpha_A, data.Ap, data.r_A)
        rel_residual_A = genblas_nrm2(data.r_A)
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

    value_d = sqrt(res_list_A[deter + 1]^2 - res_list_A[deter + 2]^2)
    first = 0
    second = 0
    second = value_d^2
    third = 0
    third = res_list_A[deter + 2]^2 - res_list_A[deter + 3]^2

    pointer1 = 0
    pointer2 = 0
    mark1 = false
    mark2 = false
    flip = 0

    for iter = deter + 2 : maxIter
        if (mark1 == true)
            first = (first * pointer1 + second) / (pointer1 + 1)
            value_d = sqrt(first)
            second = third
            third = res_list_A[iter + 1]^2 - res_list_A[iter + 2]^2
        elseif (mark2 == true)
            second = (second * pointer2 + third) / (pointer2 + 1)
            third = res_list_A[iter + 1]^2 - res_list_A[iter + 2]^2
        else
            first = second
            second = third
            third = res_list_A[iter + 1]^2 - res_list_A[iter + 2]^2           
        end


        if (first < second)
            if (flip == 0)
                mark1 = true
                pointer1 = pointer1 + 1
            else
                mark2 = true
                pointer2 = pointer2 + 1
            end
            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma_A = genblas_dot(data.r_A, data.Ar)
            alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
            #if alpha_A == Inf || alpha_A < 0
            #    return -13, iter, residual_0, res_list_B, x_B, P_list
            #end

            update_p .= update_p + alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            rel_residual_A = genblas_nrm2(data.r_A)
            res_list_A = hcat(res_list_A, rel_residual_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)
        elseif (first >= second && second < third)
            flip = 1
            mark1 = false
            mark2 = true
            pointer2 = pointer2 + 1

            A(data.Ap, data.p_A)
            A(data.Ar, data.r_A)
            gamma_A = genblas_dot(data.r_A, data.Ar)
            alpha_A = gamma_A / genblas_dot(data.Ap, data.Ap)
            #if alpha_A == Inf || alpha_A < 0
            #    return -13, iter, residual_0, res_list_B, x_B, P_list
            #end

            update_p .= update_p + alpha_A * data.p_A

            # x += alpha*p
            genblas_axpy!(alpha_A, data.p_A, x)
            # r -= alpha*Ap
            genblas_axpy!(-alpha_A, data.Ap, data.r_A)
            rel_residual_A = genblas_nrm2(data.r_A)
            res_list_A = hcat(res_list_A, rel_residual_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)
        else
            flip = 1
            mark1 = false
            mark2 = false
            weight = 1 / (1-sum_p)
            genblas_axpy!(weight, update_p, x_B)

            value_n = sqrt(first) - sqrt(second)
            p_p = value_n / value_d

            if (pointer1 > 0)
                for i = 1 : pointer1
                    P_list = hcat(P_list, 0)
                end
            end

            if (pointer2 > 0)
                for i = 1 : pointer2
                    P_list = hcat(P_list, 0)
                end
            end

            P_list = hcat(P_list, p_p)
            # Random.seed!(7 * (iter + 1 - pointer1 - pointer2) + 1 + 35 * seed)
            d = rand()

            if (d < (p_p / (1 - sum_p)))
                return x, x_B, iter+1
            end

            sum_p += p_p
            
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
            rel_residual_A = genblas_nrm2(data.r_A)
            res_list_A = hcat(res_list_A, rel_residual_A)

            #if rel_residual_A <= tol
            #    return 30, 1, rel_residual_A, res_list, x, P_list
            #end
            A(data.Ar, data.r_A)
            beta_A = genblas_dot(data.r_A, data.Ar) / gamma_A
            # p = z + beta*p
            genblas_scal!(beta_A, data.p_A)
            genblas_axpy!(1.0, data.r_A, data.p_A)

            pointer2 = 0
            pointer1 = 0
        end
    end

    return x, x_B, maxIter
end

export CGData3, resrcr_theory_squared_c3!