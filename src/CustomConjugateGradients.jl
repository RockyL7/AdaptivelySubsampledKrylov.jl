module CustomConjugateGradients

    include("genericblas.jl")
    include("reader.jl")
    include("cg.jl")
    include("bicgstab.jl")
    include("rrcg.jl")
    include("resrcg.jl")
    include("resrcg2.jl")
    include("cr.jl")
    include("resrcr_new.jl")
    include("resrcr_theory.jl")
    include("resrcr_theory_c.jl")
    include("resrcr_theory_squared.jl")
    include("resrcr_comp1.jl")
    include("resrcr_comp2.jl")
    include("resrcr_theory_squared_c.jl")
    include("resrcr_theory_squared_c2.jl")
    include("resrcr_theory_squared_c3.jl")
    include("resrcg_theory_squared_c3.jl")

end
