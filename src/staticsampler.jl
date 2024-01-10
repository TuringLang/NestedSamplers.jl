# Sampler and model implementations

struct Nested{B, P <: AbstractProposal} <: AbstractSampler
    type::Type
    ndims::Int
    nactive::Int
    bounds::B
    enlarge::Float64
    update_interval::Int
    min_ncall::Int
    min_eff::Float64
    proposal::P
    dlv::Float64
end

"""
    Nested(ndims, nactive;
        bounds=Bounds.MultiEllipsoid,
        proposal=:auto,
        enlarge=1.25,
        update_interval=default_update_interval(proposal, ndims),
        min_ncall=2nactive,
        min_eff=0.10)

Static nested sampler with `nactive` active points and `ndims` parameters.

`ndims` is equivalent to the number of parameters to fit, which defines the dimensionality of the prior volume used in evidence sampling. `nactive` is the number of live or active points in the prior volume. This is a static sampler, so the number of live points will be constant for all of the sampling.

## Bounds and Proposals

`bounds` declares the Type of [`Bounds.AbstractBoundingSpace`](@ref) to use in the prior volume. The available bounds are described by [`Bounds`](@ref). `proposal` declares the algorithm used for proposing new points. The available proposals are described in [`Proposals`](@ref). If `proposal` is `:auto`, will choose the proposal based on `ndims`
* `ndims < 10` - [`Proposals.Rejection`](@ref)
* `10 ≤ ndims ≤ 20` - [`Proposals.RWalk`](@ref)
* `ndims > 20` - [`Proposals.Slice`](@ref)

The original nested sampling algorithm is roughly equivalent to using `Bounds.Ellipsoid` with `Proposals.Rejection`. The MultiNest algorithm is roughly equivalent to `Bounds.MultiEllipsoid` with `Proposals.Rejection`. The PolyChord algorithm is roughly equivalent to using `Proposals.RSlice`.

## Other Parameters
* `enlarge` - When fitting the bounds to live points, they will be enlarged (in terms of volume) by this linear factor.
* `update_interval` - How often to refit the live points with the bounds as a fraction of `nactive`. By default this will be determined using `default_update_interval` for the given proposal
    * `Proposals.Rejection` - `1.5`
    * `Proposals.RWalk` and `Proposals.RStagger` - `0.15 * walks`
    * `Proposals.Slice` - `0.9 * ndims * slices`
    * `Proposals.RSlice` - `2 * slices`
* `min_ncall`: The minimum number of iterations before fitting the first bound; used to 
avoid shrinking the bounds before burn-in is completed. By default 2*`nactive`.
* `min_eff`: Minimum efficiency `(samples accepted / samples generated)` before fitting the 
first bound; used to avoid shrinking the bounds before burn-in is completed By default 0.1.
"""
function Nested(ndims,
    nactive;
    bounds = Bounds.MultiEllipsoid,
    proposal = :auto,
    enlarge = 1.25,
    min_ncall=2nactive,
    min_eff=0.10,
    type = Float64,
    kwargs...)

    nactive < 2ndims && @warn "Using fewer than 2ndim ($(2ndims)) active points is discouraged"

    # get proposal
    if proposal === :auto
        proposal = if ndims < 10
            Proposals.Rejection()
        elseif 10 ≤ ndims ≤ 20
            Proposals.RWalk() 
        else
            Proposals.Slice()
        end
    end

    dlv = log(nactive + 1) - log(nactive)

    update_interval_frac = get(kwargs, :update_interval, default_update_interval(proposal, ndims))
    update_interval = round(Int, update_interval_frac * nactive)
    return Nested(
        type,
        ndims,
        nactive,
        bounds,
        enlarge,
        update_interval,
        min_ncall,
        min_eff,
        proposal,
        dlv)
end

default_update_interval(p::Proposals.Rejection, ndims) = 1.5
default_update_interval(p::Proposals.RWalk, ndims) = 0.15 * p.walks
default_update_interval(p::Proposals.RStagger, ndims) = 0.15 * p.walks
default_update_interval(p::Proposals.Slice, ndims) = 0.9 * ndims * p.slices
default_update_interval(p::Proposals.RSlice, ndims) = 2.0 * p.slices
