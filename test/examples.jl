using Distributions
using AbstractMCMC

@testset "Flat" begin
    logl(x::AbstractVector) = zero(eltype(x))
    priors = [Uniform(0, 1)]
    model = NestedModel(logl, priors)

    for method in [:single, :multi]
        spl = Nested(4, method=method)
        chain = sample(model, spl, 100, param_names=["x"])

        @test spl.logz ≈ 0 atol=1e-10
        @test spl.h ≈ 0 atol=1e-10
    end
end

@testset "Gaussian" begin

end

@testset "Egg Shell" begin

end
