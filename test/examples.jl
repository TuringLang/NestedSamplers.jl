using Distributions
using AbstractMCMC

@testset "Flat" begin
    logl(x::AbstractVector) = zero(eltype(x))
    priors = [Uniform(0, 1)]
    model = NestedModel(logl, priors)

    for method in [:single, :multi]
        chain = sample(model, Nested(4, method=method), 100, param_names=["x"])
        logz = Array(chain[:logz])
        h = Array(chain[:h])

        @test logz[end] ≈ 0 atol=1e-10
        @test_broken h[end] ≈ 0 atol=1e-10
    end
end

@testset "Gaussian" begin

end

@testset "Egg Shell" begin

end
