using Distributions
using AbstractMCMC
using MCMCChains: Chains

@testset "Flat" begin
    logl(x::AbstractVector) = zero(eltype(x))
    priors = [Uniform(0, 1)]
    model = NestedModel(logl, priors)

    for method in [:single, :multi]
        spl = Nested(4, method = method)
        chain = sample(model, spl, param_names = ["x"], chain_type = Chains)

        @test spl.logz ≈ 0 atol = 1e-10
        @test spl.h ≈ 0 atol = 1e-10
    end
end

@testset "Gaussian" begin
    σ = 0.1
    μ1 = ones(2)
    μ2 = -ones(2)
    inv_σ = diagm(0 => fill(1 / σ^2, 2))

    function logl(x)
        dx1 = x .- μ1
        dx2 = x .- μ2
        f1 = -dx1' * (inv_σ * dx1) / 2
        f2 = -dx2' * (inv_σ * dx2) / 2
        return log(exp(f1) + exp(f2))
    end

    priors = [Uniform(-5, 5), Uniform(-5, 5)]
    model = NestedModel(logl, priors)
    
    # TODO get the grid logz working; this doesnt match test
    analytic_logz = log(2 * 2π * σ^2 / 100)

    for method in [:single, :multi]
        spl = Nested(100, method = method)
        chain = sample(model, spl, dlogz = 0.1, chain_type = Chains)

        @test_broken spl.logz ≈ analytic_logz atol = 4sqrt(spl.h / spl.nactive)
        @test sum(Array(chain[:weights])) ≈ 1 rtol = 1e-3
    end
end
