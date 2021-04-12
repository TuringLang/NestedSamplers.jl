```@meta
CurrentModule = NestedSamplers
```

# NestedSamplers.jl

[![GitHub](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/TuringLang/NestedSamplers.jl)
[![Build Status](https://github.com/turinglang/NestedSamplers.jl/workflows/CI/badge.svg?branch=master)](https://github.com/turinglang/NestedSamplers.jl/actions)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/N/NestedSamplers.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![Coverage](https://codecov.io/gh/turinglang/NestedSamplers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/turinglang/NestedSamplers.jl)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3950594.svg)](https://doi.org/10.5281/zenodo.3950594)

A Julian implementation of single- and multi-ellipsoidal nested sampling algorithms using the [AbstractMCMC](https://github.com/turinglang/abstractmcmc.jl) interface.

This package was heavily influenced by [`nestle`](https://github.com/kbarbary/nestle), [`dynesty`](https://github.com/joshspeagle/dynesty), and [`NestedSampling.jl`](https://github.com/kbarbary/NestedSampling.jl).


## Installation

To use the nested samplers first install this library

```julia
julia> ]add NestedSamplers
```

## Usage

The samplers are built using the [AbstractMCMC](https://github.com/turinglang/abstractmcmc.jl) interface. To use it, we need to create a [`NestedModel`](@ref).

```@example usage
using Random
using AbstractMCMC
AbstractMCMC.setprogress!(false)
Random.seed!(8452);
nothing # hide
```

```@example usage
using Distributions
using LinearAlgebra
using NestedSamplers
using StatsFuns: logaddexp

# multivariate Gaussian
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
priors = [
    Uniform(-5, 5),
    Uniform(-5, 5)
]
# or equivalently
prior_transform(X) = 10 .* X .- 5
# create the model
# or model = NestedModel(logl, prior_transform)
model = NestedModel(logl, priors);
nothing # hide
```

now, we set up our sampling using [StatsBase](https://github.com/JuliaStats/StatsBase.jl).

**Important:  the state of the sampler is returned in addition to the chain by `sample`.**

```@example usage
using StatsBase: sample, Weights

# create our sampler
# 2 parameters, 1000 active points, multi-ellipsoid. See docstring
spl = Nested(2, 1000)
# by default, uses dlogz for convergence. Set the keyword args here
# currently Chains and Array are support chain_types
chain, state = sample(model, spl; dlogz=0.2, param_names=["x", "y"])
# optionally resample the chain using the weights
chain_res = sample(chain, Weights(vec(chain["weights"])), length(chain));
```

let's take a look at the resampled posteriors

```@example usage
using StatsPlots
density(chain_res)
# analytical posterior maxima
vline!([-1, 1], c=:black, ls=:dash, subplot=1)
vline!([-1, 1], c=:black, ls=:dash, subplot=2)
```
and compare our estimate of the Bayesian (log-)evidence to the analytical value
```@example usage
analytic_logz = log(4π * σ^2 / 100)
# within 2-sigma
@assert isapprox(analytic_logz, state.logz, atol=2state.logzerr)
```

## Contributing

**Primary Author:** Miles Lucas ([@mileslucas](https://github.com/mileslucas))

Contributions are always welcome! Take a look at the [issues](https://github.com/turinglang/nestedsamplers.jl/issues) for ideas of open problems! To discuss ideas or plan contributions, open a [discussion](https://github.com/TuringLang/NestedSamplers.jl/discussions).
