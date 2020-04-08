# NestedSamplers.jl

[![Build Status](https://github.com/turinglang/NestedSamplers.jl/workflows/CI/badge.svg?branch=master)](https://github.com/turinglang/NestedSamplers.jl/actions)
[![Coverage](https://codecov.io/gh/turinglang/NestedSamplers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/turinglang/NestedSamplers.jl)

A Julian implementation of single- and multi-ellipsoidal nested sampling algorithms using the [AbstractMCMC](https://github.com/turinglang/abstractmcmc.jl) interface.

This package was heavily influenced by [`nestle`](https://github.com/kbarbary/nestle) and [`NestedSampling.jl`](https://github.com/kbarbary/NestedSampling.jl).


## Installation

To use the nested samplers first install this library

````julia

julia> ]add NestedSamplers
````




## Usage

The samplers are built using the [AbstractMCMC](https://github.com/turinglang/abstractmcmc.jl) interface. To use it, we need to create a `NestedModel`.


````julia
using NestedSamplers
using Distributions
using Random; Random.seed!(8462852)

# data: gaussian
data = 2 .* randn(100) .+ 5
logl(x) = loglikelihood(Normal(x...), data)
priors = [
    Uniform(-0, 10) # μ
    Uniform(0, 5)   # σ
]
# create the model
model = NestedModel(logl, priors);
````





now, we set up our sampling using [StatsBase](https://github.com/JuliaStats/StatsBase.jl)

````julia
using StatsBase: sample
using MCMCChains: Chains

# create our sampler
spl = Nested(100) # 100 active points; single-ellipsoid. See docstring
# by default, uses dlogz_convergence. Set the keyword args here
chain = sample(model, spl; 
               dlogz=0.5, 
               param_names=["mu", "sigma"], 
               chain_type=Chains)
````


````
Object of type Chains, with data of type 724×3×1 Array{Float64,3}

Log evidence      = 1.7439818907812824e-96
Iterations        = 1:724
Thinning interval = 1
Chains            = 1
Samples per chain = 724
internals         = weights
parameters        = sigma, mu

2-element Array{MCMCChains.ChainDataFrame,1}

Summary Statistics
  parameters    mean     std  naive_se    mcse        ess   r_hat
  ──────────  ──────  ──────  ────────  ──────  ─────────  ──────
       sigma  5.0319  1.4880    0.0553  0.0296     7.2034  1.1760
          mu  2.4937  0.8795    0.0327  0.2084  1294.5493  0.9990

Quantiles
  parameters    2.5%   25.0%   50.0%   75.0%   97.5%
  ──────────  ──────  ──────  ──────  ──────  ──────
          mu  1.0895  4.7007  5.0632  5.4433  8.5756
       sigma  0.8604  2.0098  2.2209  2.9211  4.5331
````



````julia
using StatsPlots
density(chain)
vline!([5], c=:black, ls=:dash, subplot=1)
vline!([2], c=:black, ls=:dash, subplot=2)
````


![](docs/figures/README_4_1.png)



## API/Reference


```
Nested(nactive; enlarge=1.2, update_interval=round(Int, 0.6nactive), method=:single)
```

Ellipsoidal nested sampler.

The two methods are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.

### Parameters

  * `nactive` - The number of live points.
  * `enlarge` - When fitting ellipsoids to live points, they will be enlarged (in terms of volume) by this factor.
  * `update_interval` - How often to refit the live points with the ellipsoids
  * `method` - as mentioned above, the algorithm to use for sampling. `:single` uses a single ellipsoid and follows the original nested sampling algorithm proposed in Skilling 2004. `:multi` uses multiple ellipsoids- much like the MultiNest algorithm.



---

```
NestedModel(loglike, priors::AbstractVector{<:Distribution})
```

A model for use with the `Nested` sampler.

`loglike` must be callable with a signature `loglike(::AbstractVector)::Real` where the length of the vector must match the number of parameters in your model.

`priors` are required for each variable in order to transform between a unit-sphere and parameter space. This means they must have `Distributions.cdf` and `Distributions.quantile` must be implemented.

**Note:** `loglike` is the only function used for likelihood calculations. This means if you want your priors to be used for the likelihood calculations they must be manually included in that function.



---

```
dlogz_convergence(args...; dlogz=0.5, kwargs...)
```

Stopping criterion: estimated fraction evidence remaining below threshold.

The estimated fraction evidence remaining is given by the `maximum(active_loglike) - it/nactive` where `it` is the current iteration.



---

```
decline_convergence(args...; decline_factor=6, kwargs...)
```

Stopping criterion: Number of consecutive declining log-evidence is greater than `iteration / decline_factor` or greater than `2nactive`




## Contributing
**Primary Author:** Miles Lucas ([@mileslucas](https://github.com/mileslucas))

Contributions are always welcome! Take a look at the [issues](https://github.com/turinglang/nestedsamplers.jl/issues) for ideas of open problems!

---

This file was generated from [README.jmd](docs/README.jmd) using [Weave.jl](https://github.com/JunoLab/Weave.jl)
