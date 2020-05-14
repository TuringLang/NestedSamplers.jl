
"""
    NestedModel(loglike, prior_transform)
    NestedModel(loglike, priors::AbstractVector{<:Distribution})

`loglike` must be callable with a signature `loglike(::AbstractVector)` where the length of the vector must match the number of parameters in your model.

`prior_transform` must be a callable with a signature `prior_transform(::AbstractVector)` that returns the transformation from the unit-cube to parameter space. This is effectively the quantile or ppf of a statistical distribution. For convenience, if a vector of `Distribution` is provided (as a set of priors), a transformation function will automatically be constructed using `Distributions.quantile`.

**Note:**
`loglike` is the only function used for likelihood calculations. This means if you want your priors to be used for the likelihood calculations they must be manually included in the `loglike` function.
"""
struct NestedModel <: AbstractModel
    loglike::Function
    prior_transform::Function
end

function NestedModel(loglike, priors::AbstractVector{<:UnivariateDistribution})
    prior_transform(X) = quantile.(priors, X)
    return NestedModel(loglike, prior_transform)
end
