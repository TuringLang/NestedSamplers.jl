using StatsBase
using StatsFuns
using NestedSamplers
using Distributions
using LinearAlgebra

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

priors = [Uniform(-5, 5), Uniform(-5, 5)]
model = NestedModel(logl, priors)

spl = Nested(10)
chain = sample(model, spl, dlogz = 0.1, chain_type = Array)
