using Distributions: Distribution, quantile, cdf
import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples, AbstractModel

export NestedModel, Nested


"""
    Nested(nactive, enlarge; update_interval=round(0.6nactive), method=:single)

Nested Sampler

The two `NestedAlgorithm`s are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.
"""
struct Nested <: AbstractSampler 
    nactive::Integer
    enlarge::Float64
    update_interval::Integer
    ell_type::Type{<:AbstractEllipsoid}
end

function Nested(nactive = 100, enlarge = 1.2; update_interval=round(Int, 0.6nactive), method=:single)
    if method === :single
        E = Ellipsoid
    elseif method === :multi
        E = MultiEllipsoid
    else
        error("Invalid method $method")
    end

    return Nested(nactive, enlarge, update_interval, E)
end

struct NestedModel{F <: Function,D <: Distribution} <: AbstractModel
    loglike::F
    priors::Vector{D}
end

struct NestedTransition{T,E<:AbstractEllipsoid} <: AbstractTransition
    active_points::Matrix{T}
    active_logl::Vector{T}
    samples::Vector{T}
    log_vol
    log_wt
    log_z
    h
    bounding_ell::E
    it::Integer
end

function NestedTransition(rng::AbstractRNG, model::NestedModel, nactive)
    ndim = length(model.priors)
    us = rand(rng, ndim, nactive)
    ps = quantile.(hcat(model.priors), us)
    return NestedTransition(model, ps)
end

NestedTransition(model::NestedModel, nactive) = NestedTransition(Random.GLOBAL_RNG, model, nactive)

function NestedTransition(model::NestedModel, p::Matrix)
    # Get info from uniform space into prior space
    logls = [model.loglike(p[:, i]) for i in 1:size(p, 2)]

    # log prior volume
    logv = log(1 - exp(-1 / size(p, 2)))

    # log evidence
    logl_star, mindx = findmin(logls)

    # samples from least_likely
    s = p[:, mindx]

    # Dummy unit ellipsoid to start
    ell = Ellipsoid(size(p, 1))

    return NestedTransition(p, logls, s, logv, -Inf, -Inf, 0, ell, 0)
end

transition_type(model::NestedModel, spl::Nested) = NestedTransition

function step!(rng::AbstractRNG,
    model::NestedModel,
    spl::Nested,
    N::Integer;
    kwargs...)
    spl.nactive < 2length(model.priors) && @warn "Using fewer than 2*ndim active points is discouraged"
    return NestedTransition(rng, model, spl.nactive)
end

function propose(rng::AbstractRNG, ell::AbstractEllipsoid, model::NestedModel, logl_star)
    while true
        u = rand(rng, ell)
        all(0 .< u .< 1) || continue
        v = quantile.(model.priors, u)
        logl = model.loglike(v)
        if logl > logl_star
            return v, logl
        end
    end
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    spl::Nested,
    N::Integer,
    prev::NestedTransition;
    kwargs...)
    # TODO put this part into a function
    # We need to flush the remaining points in the ellipse at N-nactive
    if get(kwargs, :iteration, NaN) > N - spl.nactive
        log_vol = -kwargs[:iteration] / spl.nactive - log(spl.nactive)
        i = spl.nactive - N + kwargs[:iteration]
        logl_star = prev.active_logl[i]
        log_wt = log_vol + logl_star
        logz_new = log(exp(prev.log_z) + exp(log_wt))
        h = (exp(log_wt - logz_new) * logl_star + 
            exp(prev.log_z - logz_new) * (prev.h + prev.log_z) - logz_new)
        h = isnan(h) ? prev.h : h

        samples = prev.active_points[:, i]
        
        return NestedTransition(prev.active_points, prev.active_logl, samples, prev.log_vol, log_wt, logz_new, h, prev.bounding_ell, prev.it)
    end

    # TODO put this stuff into a function

    logl_star, mindx = findmin(prev.active_logl)
    log_wt = prev.log_vol + logl_star

    logz_new = log(exp(prev.log_z) + exp(log_wt))

    h = (exp(log_wt - logz_new) * logl_star + 
        exp(prev.log_z - logz_new) * (prev.h + prev.log_z) - logz_new)
    h = isnan(h) ? prev.h : h
    
    samples = prev.active_points[:, mindx]

    # Get points in unit space
    u = cdf.(hcat(model.priors), prev.active_points)

    # Get bounding ellipsoid (only every update_interval)
    if prev.it % spl.update_interval == 0
        pointvol = exp(-get(kwargs, :iteration, 0) / spl.nactive) / spl.nactive
        ell = fit(spl.ell_type, u, pointvol=pointvol)
        scale!(ell, spl.enlarge)
        it = 0
    else
        ell = prev.bounding_ell
        it = prev.it + 1
    end

    p, logl = propose(rng, ell, model, logl_star)

    # prepare new state
    log_vol = prev.log_vol - 1 / spl.nactive
    newp = prev.active_points
    newp[:, mindx] = p
    newlogl = prev.active_logl
    newlogl[mindx] = logl


    return NestedTransition(newp, newlogl, samples, log_vol, log_wt, logz_new, h, ell, it)
end

function bundle_samples(rng::AbstractRNG, 
    ℓ::AbstractModel, 
    s::Nested, 
    N::Integer, 
    ts::Vector{<:AbstractTransition}; 
    param_names = missing,
    kwargs...)
    vals = copy(reduce(hcat, [vcat(t.samples, t.log_wt, t.log_z, t.h) for t in ts])')
    # update weights based on best evidence
    @. vals[:, end-2, 1] = exp(vals[:, end-2, 1] - vals[end, end-1, 1])

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 3]
    end
    push!(param_names, "weights", "logz", "h")

    return Chains(vals, param_names, (internals = ["weights", "logz", "h"],))
end
