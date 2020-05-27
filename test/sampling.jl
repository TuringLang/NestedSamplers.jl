using Distributions
using AbstractMCMC
using MCMCChains: Chains
using StatsFuns
using StatsBase

@testset "Bundles" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chain = sample(model, spl; dlogz = 0.2, param_names = ["x"], chain_type = Chains, progress=false)
    samples = sample(model, spl; dlogz = 0.2, chain_type = Array, progress=false)
end

# @testset "Flat" begin
#     logl(x::AbstractVector{T}) where T = zero(T)
#     priors = [Uniform(0, 1)]
#     model = NestedModel(logl, priors)

#     for method in [:single, :multi]
#         spl = Nested(4, method = method)
#         chain = sample(model, spl, dlogz = 0.2, chain_type = Array)

#         @test spl.logz ≈ 0 atol = 1e-9 # TODO
#         @test spl.h ≈ 0 atol = 1e-9 # TODO
#     end
# end

@testset "Gaussian - $bound, $P" for bound in [Bounds.NoBounds, Bounds.Ellipsoid, Bounds.MultiEllipsoid],
                        P in [Proposals.Uniform, Proposals.RWalk]
    σ = 0.1
    μ1 = ones(2)
    μ2 = -ones(2)
    inv_σ = diagm(0 => fill(1 / σ^2, 2))

    function logl(x)
        dx1 = x .- μ1
        dx2 = x .- μ2
        f1 = -dx1' * (inv_σ * dx1) / 2
        f2 = -dx2' * (inv_σ * dx2) / 2
        return logaddexp(f1, f2)
    end

    prior(X) = 10 .* X .- 5
    model = NestedModel(logl, prior)
    
    analytic_logz = log(4π * σ^2 / 100)

    
    spl = Nested(2, 1000, bounds = bound, proposal = P())
    chain = sample(model, spl, dlogz=0.1, chain_type = Array, progress=false)
    # TODO why does RWalk do poorly here ;()
    if P <: Proposals.Uniform
        @test spl.logz ≈ analytic_logz atol = 3sqrt(spl.h / spl.nactive) # within 3σ
        @test sort!(findpeaks(chain[:, 1, 1])[1:2]) ≈ [-1, 1] rtol = 3e-2
        @test sort!(findpeaks(chain[:, 2, 1])[1:2]) ≈ [-1, 1] rtol = 3e-2
    end
end
