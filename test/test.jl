using StatsBase
using NestedSamplers
using Distributions

logl(x) = 0
priors = [Uniform(0, 1)]
model = NestedModel(logl, priors)

spl = Nested(4)
chain = sample(model, spl, 10; chain_type = Array)
