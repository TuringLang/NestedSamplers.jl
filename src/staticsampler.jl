# Sampler and model implementations

mutable struct Nested{T,B <: AbstractBoundingSpace{T},P <: AbstractProposal} <: AbstractSampler
    ndims::Int
    nactive::Int
    enlarge::Float64
    update_interval::Int
    min_ncall::Int
    min_eff::Float64
    has_bounds::Bool
    active_us::Matrix{T}
    active_points::Matrix{T}
    active_logl::Vector{T}
    active_bound::B
    proposal::P
    logz::Float64
    h::Float64
    log_vol::Float64
    ndecl::Int
    ncall::Int
end

"""
    Nested(ndims, nactive;
        bounds=Bounds.MultiEllipsoid,
        proposal=:auto,
        enlarge=1.25,
        update_interval=default_update_interval(proposal),
        min_ncall=2nactive,
        min_eff=0.10)

Static nested sampler with `nactive` active points and `ndims` parameters.

`ndims` is equivalent to the number of parameters to fit, which defines the dimensionality of the prior volume used in evidence sampling. `nactive` is the number of live or active points in the prior volume. This is a static sampler, so the number of live points will be constant for all of the sampling.

## Bounds and Proposals

`bounds` declares the Type of [`Bounds.AbstractBoundingSpace`](@ref) to use in the prior volume. The available bounds are described by [`Bounds`](@ref). `proposal` declares the algorithm used for proposing new points. The available proposals are described in [`Proposals`](@ref). If `proposal` is `:auto`, will choose the proposal based on `ndims`
* `ndims < 10` - [`Proposals.Uniform`](@ref)
* `10 <= ndims <= 20` - [`Proposals.RWalk`](@ref) or [`Proposals.RStagger`](@ref)

The original nested sampling algorithm is roughly equivalent to using `Bounds.Ellipsoid` with `Proposals.Uniform`. The MultiNest algorithm is roughly equivalent to `Bounds.MultiEllipsoid` with `Proposals.Uniform`.

## Other Parameters
* `enlarge` - When fitting the bounds to live points, they will be enlarged (in terms of volume) by this linear factor.
* `update_interval` - How often to refit the live points with the bounds as a fraction of `nactive`. By default this will be determined using `default_update_interval` for the given proposal
    * For `Proposals.Uniform` - `1.5`
    * For `Proposals.RWalk` and `Proposals.RStagger` - `0.15walks`
* `min_ncall` - The minimum number of iterations before trying to fit the first bound
* `min_eff` - The maximum efficiency before trying to fit the first bound
"""
function Nested(ndims,
    nactive;
    bounds = Bounds.Ellipsoid,
    proposal = :auto,
    enlarge = 1.25,
    min_ncall=2nactive,
    min_eff=0.10,
    kwargs...)

    nactive < 2ndims && @warn "Using fewer than 2ndim ($(2ndims)) active points is discouraged"

    # get proposal
    if proposal === :auto
        proposal = if ndims < 10
            Proposals.Uniform()
        else#if 10 ≤ ndims ≤ 20
            Proposals.RWalk() # _Proposals.RStagger()_ can be taken as an alternative to _Proposals.RWalk_ 
            # for ndims > 20, _Proposals.HSlice_ method if _gradient_ is provided, else _Proposals.Slice_ method to be used (this will be included later on)  
        end
    end

    update_interval_frac = get(kwargs, :update_interval, default_update_interval(proposal))
    update_interval = round(Int, update_interval_frac * nactive)
    B = bounds(ndims)
    # Initial point will have volume 1 - exp(-1/npoints)
    log_vol = log1mexp(-1 / nactive)
    #= Note: initializing logz as -Inf causes ugly failures in the h calculations
    by setting to a very small value (even smaller than log(eps(Float64))) we avoid this issue =#
    return Nested(ndims,
        nactive,
        enlarge,
        update_interval,
        min_ncall,
        min_eff,
        false,
        zeros(ndims, nactive),
        zeros(ndims, nactive),
        zeros(nactive),
        B,
        proposal,
        -1e300,
        0.0,
        log_vol,
        0,
        0)
end

default_update_interval(p::Proposals.Uniform) = 1.5
default_update_interval(p::Proposals.RWalk) = 0.15p.walks
default_update_interval(p::Proposals.RStagger) = 0.15p.walks


function Base.show(io::IO, n::Nested)
    println(io, "Nested(ndims=$(n.ndims), nactive=$(n.nactive), enlarge=$(n.enlarge), update_interval=$(n.update_interval))")
    println(io, "  bounds=$(n.active_bound)")
    println(io, "  proposal=$(n.proposal)")
    println(io, "  logz=$(n.logz)")
    println(io, "  log_vol=$(n.log_vol)")
    print(io,   "  H=$(n.h)")
end

struct NestedTransition{T}
    draw::Vector{T}  # the sample
    logL::Float64    # log likelihood
    log_wt::Float64  # log weight of this draw
end

function Base.show(io::IO, t::T) where {T <: NestedTransition}
    println(io, "$T")
    println(io, "  $(t.draw)")
    println(io, "  log-likelihood=$(t.logL)")
    print(io,   "  log-weight=$(t.log_wt)")
end
