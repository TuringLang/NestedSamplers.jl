"""
This module contains various statistical models in the form of [`NestedModel`](@ref)s. These models can be used for examples and for testing.

* [`Models.CorrelatedGaussian`](@ref)
"""
module Models

using ..NestedSamplers

using Distributions
using LinearAlgebra

@doc raw"""
    Models.CorrelatedGaussian(ndims)

Creates a highly-correlated Gaussian with the given dimensionality.

```math
\mathbf\theta \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)
```
```math
\Sigma_{ij} = \begin{cases} 1 &\quad i=j \\ 0.95 &\quad i\neq j \end{cases}
```
```math
\mathcal{L}(\mathbf\theta) = \mathcal{N}\left(\mathbf\theta | 2\mathbf{1}, \mathbf\Sigma \right)
```

the analytical evidence of the model is

```math
Z = \mathcal{N}\left(2\mathbf{1} | \mathbf{0}, \mathbf\Sigma + \mathbf{I} \right)
```

## Examples
```jldoctest
julia> model, lnZ = Models.CorrelatedGaussian(10);

julia> lnZ
-12.482738597926607
```
"""
function CorrelatedGaussian(ndims)
    priors = fill(Normal(), ndims)
    μ = fill(2.0, ndims)
    Σ = fill(0.95, ndims, ndims)
    Σ[diagind(Σ)] .= 1
    cent_dist = MvNormal(μ, Σ)
    loglike(X) = logpdf(cent_dist, X)

    model = NestedModel(loglike, priors)
    true_lnZ = logpdf(MvNormal(Σ + I), μ)
    return model, true_lnZ
end

end # module
