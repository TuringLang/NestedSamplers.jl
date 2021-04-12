# API/Reference

```@index
```

## Samplers

```@docs
NestedModel
Nested
```

### Convergence

 There are a few convergence criteria available, by default the `dlogz` criterion will be used.
* `dlogz=0.5` sample until the *fraction of the remaining evidence* is below the given value ([more info](https://dynesty.readthedocs.io/en/latest/overview.html#stopping-criteria)).
* `maxiter=Inf` stop after the given number of iterations
* `maxcall=Inf` stop after the given number of  log-likelihood function calls
* `maxlogl=Inf` stop after reaching the target log-likelihood

## Bounds

```@docs
Bounds
Bounds.NoBounds
Bounds.Ellipsoid
Bounds.MultiEllipsoid
```

## Proposals

```@docs
Proposals
Proposals.Uniform
Proposals.RWalk
Proposals.RStagger
Proposals.Slice
Proposals.RSlice
```

## Models

```@docs
Models
Models.CorrelatedGaussian
```