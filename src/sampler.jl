using Distributions: Distribution, quantile, cdf
import MCMCChains: Chains
import AbstractMCMC: AbstractSampler, AbstractTransition, AbstractModel, step!, sample_init!, sample_end!, transition_type, bundle_samples

export NestedModel, Nested

mutable struct Nested{E<:AbstractEllipsoid} <: AbstractSampler 
    nactive         ::Integer
    enlarge         ::Float64
    update_interval ::Integer
    # behind the scenes things
    active_points   ::Matrix
    active_logl     ::Vector
    active_ell      ::E
    logz            ::Float64
    h               ::Float64
end

"""
    Nested(nactive, enlarge; update_interval=round(0.6nactive), method=:single)

Nested Sampler

The two `NestedAlgorithm`s are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.
"""
function Nested(nactive = 100, enlarge = 1.2; update_interval=round(Int, 0.6nactive), method=:single)
    if method === :single
        ell = Ellipsoid(1)
    elseif method === :multi
        ell = MultiEllipsoid([Ellipsoid(1)])
    else
        error("Invalid method $method")
    end

    return Nested(nactive, enlarge, update_interval, zeros(0,nactive), zeros(nactive), ell, -Inf, 0.0)
end

Base.show(io::IO, n::Nested) = print(io, "Nested{$(typeof(n.active_ell))}(nactive=$(n.nactive), enlarge=$(n.enlarge), update_interval=$(n.update_interval))")

struct NestedModel{F <: Function,D <: Distribution} <: AbstractModel
    loglike::F
    priors::Vector{D}
end

struct NestedTransition <: AbstractTransition
    draw::Vector
    logL # log likelihood
    log_vol # log mass of objects in hyperspace
    log_wt # log weight of this draw
end

transition_type(model::NestedModel, s::Nested) = NestedTransition

function sample_init!(
    rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    N::Integer;
    debug::Bool=false,
    kwargs...
) where {E<:AbstractEllipsoid}
    debug && @info "Initializing sampler"
    s.nactive < 2length(model.priors) && @warn "Using fewer than 2*ndim active points is discouraged"

    # samples in unit space
    us = rand(rng, length(model.priors), s.nactive)

    # samples and loglikes in prior space
    s.active_points = quantile.(hcat(model.priors), us)
    s.active_logl = [model.loglike(s.active_points[:, i]) for i in 1:s.nactive]

    # get bounding ellipsoid
    s.active_ell = scale!(fit(E, us, pointvol=1/s.nactive), s.enlarge)
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested,
    N::Integer;
    kwargs...)
    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]

    # Initial point will have volume 1 - exp(-1/npoints)
    log_vol = log(1 - exp(-1/s.nactive))
    log_wt = log_vol + logL

    return NestedTransition(draw, logL, log_vol, log_wt)
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    N::Integer,
    prev::NestedTransition;
    iteration,
    kwargs...
    ) where {E<:AbstractEllipsoid}

    # update sampler
    logz = log(exp(s.logz) + exp(prev.log_wt))
    s.h = (exp(prev.log_wt - logz) * prev.logL + 
            exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    # Get bounding ellipsoid (only every update_interval)
    if iteration % s.update_interval == 0
        # Get points in unit space
        u = cdf.(hcat(model.priors), s.active_points)

        # fit ellipsoid
        pointvol = exp(-(iteration - 1) / s.nactive) / s.nactive
        s.active_ell = scale!(fit(E, u, pointvol=pointvol), s.enlarge)
    end

    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]

    log_vol = prev.log_vol - 1 / s.nactive
    log_wt = log_vol + logL

    # Get new point and log like
    p, logl = propose(rng, s.active_ell, model, logL)
    s.active_points[:, idx] = p
    s.active_logl[idx] = logl

    return NestedTransition(draw, logL, log_vol, log_wt)
end

function sample_end!(
    rng::AbstractRNG,
    ℓ::AbstractModel,
    s::Nested,
    N::Integer,
    ts::Vector{<:AbstractTransition};
    debug::Bool=false,
    kwargs...
)
    debug && @info "Flushing remaining $(s.nactive) points"

    prev = ts[end]
    log_vol = -N / s.nactive - log(s.nactive)
    for i in 1:s.nactive
        # update sampler
        logz = log(exp(s.logz) + exp(prev.log_wt))
        s.h = (exp(prev.log_wt - logz) * prev.logL + 
            exp(s.logz - logz) * (s.h + s.logz) - logz)
        s.logz = logz

        # get new point
        draw = s.active_points[:, i]
        logL = s.active_logl[i]
        log_wt = log_vol + logL

        prev = NestedTransition(draw, logL, log_vol, log_wt)
        push!(ts, prev)
    end

    # h should always be non-negative. Numerical error can arise from pathological corner cases
    if s.h < 0.0
        s.h = s.h > -√eps(s.h) ? zero(s.h) : @warn "Negative h encountered h=$(s.h). This is likely a bug"
    end

    println(stderr, "logz=$(s.logz) +- $(sqrt(s.h / s.nactive))")
    println(stderr, "h=$(s.h)")
end

function bundle_samples(rng::AbstractRNG, 
    ℓ::AbstractModel, 
    s::Nested, 
    N::Integer, 
    ts::Vector{<:AbstractTransition}; 
    param_names = missing,
    kwargs...)
    vals = copy(reduce(hcat, [vcat(t.draw, t.log_wt) for t in ts])')
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - s.logz)

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, (internals = ["weights"],), evidence=exp(logz))
end

function propose(rng::AbstractRNG, ell::AbstractEllipsoid, model::NestedModel, logl_star)
    while true
        u = rand(rng, ell)
        all(0 .< u .< 1) || continue
        v = quantile.(model.priors, u)
        logl = model.loglike(v)
        if logl ≥ logl_star
            return v, logl
        end
    end
end
