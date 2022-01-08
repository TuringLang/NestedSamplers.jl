
# NestedSamplers.jl

[![Build Status](https://github.com/TuringLang/NestedSamplers.jl/workflows/CI/badge.svg?branch=main)](https://github.com/TuringLang/NestedSamplers.jl/actions)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/N/NestedSamplers.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)
[![Coverage](https://codecov.io/gh/TuringLang/NestedSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/NestedSamplers.jl)
[![LICENSE](https://img.shields.io/github/license/TuringLang/NestedSamplers.jl?color=yellow)](LICENSE)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.github.io/NestedSamplers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.github.io/NestedSamplers.jl/dev)

Implementations of single- and multi-ellipsoidal nested sampling algorithms in pure Julia. We implement the [AbstractMCMC.jl](https://github.com/TuringLang/abstractmcmc.jl) interface, allowing straightforward sampling from a variety of statistical models.

This package was heavily influenced by [nestle](https://github.com/kbarbary/nestle), [dynesty](https://github.com/joshspeagle/dynesty), and [NestedSampling.jl](https://github.com/kbarbary/NestedSampling.jl).

## Citing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3950594.svg)](https://doi.org/10.5281/zenodo.3950594)

If you use this library, or a derivative of it, in your work, please consider citing it. This code is built off a multitude of academic works, which have been noted in the docstrings where appropriate. These references, along with references for the more general calculations, can all be found in [CITATION.bib](CITATION.bib)

## Installation

To use the nested samplers first install this library

```julia
julia> ]add NestedSamplers
```

## Usage

For in-depth usage, see the [online documentation](https://TuringLang.github.io/NestedSamplers.jl/dev/). In general, you'll need to write a log-likelihood function and a prior transform function. These are supplied to a `NestedModel`, defining the statistical model

```julia
using NestedSamplers
using Distributions
using LinearAlgebra

logl(X) = logpdf(MvNormal([1, -1], I), X)
prior(X) = 4 .* (X .- 0.5)
# or equivalently
priors = [Uniform(-2, 2), Uniform(-2, 2)]
model = NestedModel(logl, priors)
```

after defining the model, set up the nested sampler. This will involve choosing the bounding space and proposal scheme, or you can rely on the defaults. In addition, we need to define the dimensionality of the problem and the number of live points. More points results in a more precise evidence estimate at the cost of runtime. For more information, see the docs.

```julia
bounds = Bounds.MultiEllipsoid
prop = Proposals.Slice(slices=10)
# 1000 live points
sampler = Nested(2, 1000; bounds=bounds, proposal=prop)
```

once the sampler is set up, we can leverage all of the [AbstractMCMC](https://github.com/turinglang/abstractmcmc.jl) interface, including the step iterator, transducer, and a convenience `sample` method. The `sample` method takes keyword arguments for the convergence criteria.

**Note:** both the samples *and* the sampler state will be returned by `sample`

```julia
using StatsBase
chain, state = sample(model, sampler; dlogz=0.2)
```

you can resample taking into account the statistical weights, again using StatsBase

```julia
chain_resampled = sample(chain, Weights(vec(chain["weights"])), length(chain))
```

These are chains from [MCMCChains](https://github.com/turinglang/mcmcchains.jl), which offer a lot of flexibility in exploring posteriors, combining data, and offering lots of convenient conversions (like to `DataFrame`s).

Finally, we can see the estimate of the Bayesian evidence

```julia
using Measurements
state.logz ± state.logzerr
```

## Contributions and Support

**Primary Author:** Miles Lucas ([@mileslucas](https://github.com/mileslucas))

Contributions are always welcome! Take a look at the [issues](https://github.com/TuringLang/NestedSamplers.jl/issues) for ideas of open problems! To discuss ideas or plan contributions, open a [discussion](https://github.com/TuringLang/NestedSamplers.jl/discussions).
