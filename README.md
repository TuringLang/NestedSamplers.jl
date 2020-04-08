


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

## API/Reference


```
Nested(nactive, enlarge=1.2; update_interval=round(Int, 0.6nactive), method=:single)
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
