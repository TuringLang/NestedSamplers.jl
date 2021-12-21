struct PriorTransformAndLogLikelihood{T,L}
    prior_transform::T
    loglikelihood::L
end

function (f::PriorTransformAndLogLikelihood)(u)
    v = f.prior_transform(u)
    return (v, f.loglikelihood(v))
end

prior_transform(f::PriorTransformAndLogLikelihood, u) = f.prior_transform(u)
function loglikelihood_from_uniform(f::PriorTransformAndLogLikelihood, u)
    return last(prior_transform_and_loglikelihood(f, u))
end
prior_transform_and_loglikelihood(f::PriorTransformAndLogLikelihood, u) = f(u)

"""
    NestedModel(loglike, prior_transform)
    NestedModel(loglike, priors::AbstractVector{<:Distribution})

`loglike` must be callable with a signature `loglike(::AbstractVector)` where the length of the vector must match the number of parameters in your model.

`prior_transform` must be a callable with a signature `prior_transform(::AbstractVector)` that returns the transformation from the unit-cube to parameter space. This is effectively the quantile or ppf of a statistical distribution. For convenience, if a vector of `Distribution` is provided (as a set of priors), a transformation function will automatically be constructed using `Distributions.quantile`.

**Note:**
`loglike` is the only function used for likelihood calculations. This means if you want your priors to be used for the likelihood calculations they must be manually included in the `loglike` function.
"""
struct NestedModel{F} <: AbstractModel
    prior_transform_and_loglike::F
end

function NestedModel(loglike, prior_transform)
    return NestedModel(PriorTransformAndLogLikelihood(prior_transform, loglike))
end

function NestedModel(loglike, priors::AbstractVector{<:UnivariateDistribution})
    prior_transform(X) = quantile.(priors, X)
    return NestedModel(loglike, prior_transform)
end

function prior_transform(model, args...)
    return first(prior_transform_and_loglikelihood(model, args...))
end

function prior_transform(model::NestedModel{<:PriorTransformAndLogLikelihood}, args...)
    return prior_transform(model.prior_transform_and_loglike, args...)
end

function loglikelihood(model, args...)
    return last(prior_transform_and_loglikelihood(model, args...))
end

function loglikelihood(model::NestedModel{<:PriorTransformAndLogLikelihood}, args...)
    return loglikelihood(model.prior_transform_and_loglike, args...)
end

function prior_transform_and_loglikelihood(model::NestedModel, args...)
    return model.prior_transform_and_loglike(args...)
end
