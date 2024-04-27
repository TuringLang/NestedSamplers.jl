module NestedSamplersDynamicPPLExt

if isdefined(Base, :get_extension)
    using NestedSamplers: NestedSamplers
    using DynamicPPL: DynamicPPL, Bijectors, Distributions
else
    using ..NestedSamplers: NestedSamplers
    using ..DynamicPPL: DynamicPPL, Bijectors, Distributions
end


"""
    find_segments(xs)

Find the indices of the last element of each segment of `xs`,
returning a vector of indices representing the end of each segment.

!!! warning
    `xs` must be sorted.

# Examples
```julia
julia> find_segments([1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
3-element Array{Int64,1}:
  3
  5
 10
```
"""
function find_segments(xs)
    segments = Int[]
    for i in 1:length(xs)
        if i == 1
            continue
        elseif xs[i] != xs[i-1]
            push!(segments, i - 1)
        end
    end
    push!(segments, length(xs))
    return segments
end

function ranges_from_segments(end_indices)
    offset = 0
    rs = UnitRange{Int}[]
    for end_index in end_indices
        r = offset + 1:end_index
        push!(rs, r)
        offset = r[end]
    end
    return rs
end

function ranges_from_lengths(lengths)
    offset = 0
    rs = UnitRange{Int}[]
    for l in lengths
        r = offset + 1:offset + l
        push!(rs, r)
        offset = r[end]
    end
    return rs
end


# Transforming distributions.
from_unit_cube_transform(x) = Bijectors.inverse(to_unit_cube_transform(x))
function to_unit_cube_transform(dist::Distributions.UnivariateDistribution)
    lb, ub = minimum(dist), maximum(dist)
    if !(isfinite(lb) && isfinite(ub))
        throw(ArgumentError("Cannot transform to unit cube if lower and upper bounds are not finite."))
    end

    # Transform to unit cube.
    to_unitcube = (
        Bijectors.Shift(-eltype(dist)(1))         # ↦ [-1, 1]
        ∘ Bijectors.Scale(2 * inv(ub - lb))       # ↦ [0, 2]
        ∘ Bijectors.Shift(-lb)                    # ↦ [0, ub - lb]
    )
    return to_unitcube
end

function to_unit_cube_transform(dist::Distributions.Product)
    # `Product` is for univariate distributions.
    dists = dist.v
    end_indices = find_segments(dists)
    rs = ranges_from_segments(end_indices)
    return Bijectors.Stacked(map(to_unit_cube_transform, dists[end_indices]), rs)
end

# For models.
const UnivariateOrMultivariateDistribution = Union{
    Distributions.UnivariateDistribution,
    Distributions.MultivariateDistribution
}

function to_unit_cube_transform(model::DynamicPPL.Model)
    dists = values(DynamicPPL.extract_priors(model))
    fs = map(to_unit_cube_transform, dists)
    input_lengths = map(length, dists)
    if any(!Base.Fix2(isa, UnivariateOrMultivariateDistribution), dists)
        throw(ArgumentError(
            "Transform to unit cube for distributions which are not " *
            "univariate or multivariate is currently not supported."
        ))
    end  
    
    return Bijectors.Stacked(fs, ranges_from_lengths(input_lengths))
end

struct TuringNestedModel{M<:DynamicPPL.Model,V<:DynamicPPL.AbstractVarInfo,F}
    model::M
    varinfo::V
    from_transform::F
end

TuringNestedModel(model, varinfo=DynamicPPL.VarInfo(model)) = TuringNestedModel(model, varinfo, from_unit_cube_transform(model))

function logl(model::TuringNestedModel, x)
    return DynamicPPL.loglikelihood(
        model.model,
        DynamicPPL.unflatten(model.varinfo, x)
    )
end

function NestedSamplers.NestedModel(model::DynamicPPL.Model; kwargs...)
    return NestedSamplers.NestedModel(TuringNestedModel(model); kwargs...)
end
function NestedSamplers.NestedModel(model::TuringNestedModel; kwargs...)
    return NestedSamplers.NestedModel(
        Base.Fix1(logl, model),
        model.from_transform,;
        kwargs...
    )
end

end
