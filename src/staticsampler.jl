# Sampler and model implementations

mutable struct Nested{T,B <: AbstractBoundingSpace{T},P <: AbstractProposal} <: AbstractSampler
    ndims::Int
    nactive::Int
    enlarge::Float64
    update_interval::Int
    active_us::Matrix{T}
    active_points::Matrix{T}
    active_logl::Vector{T}
    active_bound::B
    proposal::P
    logz::Float64
    h::Float64
    log_vol::Float64
    ndecl::Int
end

"""
    Nested(ndims, nactive; enlarge=1.2, update_interval=round(Int, 0.6nactive), method=:single)

Ellipsoidal nested sampler.

The two methods are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.

### Parameters
* `nactive` - The number of live points.
* `enlarge` - When fitting ellipsoids to live points, they will be enlarged (in terms of volume) by this factor.
* `update_interval` - How often to refit the live points with the ellipsoids
* `method` - as mentioned above, the algorithm to use for sampling. `:single` uses a single ellipsoid and follows the original nested sampling algorithm proposed in Skilling 2004. `:multi` uses multiple ellipsoids- much like the MultiNest algorithm.
"""
function Nested(ndims,
    nactive;
    enlarge = 1.2,
    update_interval = round(Int, 0.6nactive),
    bounds = Bounds.Ellipsoid,
    proposal = Proposals.Uniform())
    nactive < 2ndims && @warn "Using fewer than 2ndim ($(2ndims)) active points is discouraged"
    B = bounds(ndims)
    # Initial point will have volume 1 - exp(-1/npoints)
    log_vol = log1mexp(-1 / nactive)
    #= Note: initializing logz as -Inf causes ugly failures in the h calculations
    by setting to a very small value (even smaller than log(eps(Float64))) we avoid this issue =#
    return Nested(ndims, nactive, enlarge, update_interval, zeros(ndims, nactive), zeros(ndims, nactive), zeros(nactive), B, proposal, -1e300, 0.0, log_vol, 0)
end

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
