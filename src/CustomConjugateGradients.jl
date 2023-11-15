module CustomConjugateGradients

    include("genericblas.jl")
    include("reader.jl")
    include("cg.jl")
    include("rrcg.jl")
    include("rrcr.jl")
    include("cr.jl")
    include("resrcr_theory_squared_c3.jl")
    include("resrcg_theory_squared_c4.jl")
end
