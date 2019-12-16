using Distributions, AbstractMCMC, NestedSamplers

data = rand(Normal(5, 3), 1000)
loglike(p) = loglikelihood(Normal(p...), data)
priors = [Uniform(0, 10), Uniform(0, 5)]

spl = Nested()
model = NestedModel(loglike, priors)

chain = sample(model, spl, 1000; param_names=["mu", "sigma"])
