using Distributions: Distribution, quantile, cdf
using Random
import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples, AbstractModel

export NestedModel, Nested


"""
    Nested(nactive, enlarge, method=:single)

Nested Sampler

The two `NestedAlgorithm`s are `:single`, which uses a single bounding ellipsoid, and `:multi`, which finds an optimal clustering of ellipsoids.
"""
struct Nested <: AbstractSampler 
    nactive::Integer
    enlarge::Float64
    method::Function
end

function Nested(nactive = 100, enlarge = 1.5; method=:single)
    if method === :single
        m(x, e) = fit(Ellipsoid, x, e)
    elseif method === :multi
        error("Not implemented")
    else
        error("Invalid method $method")
    end
    Nested(nactive, enlarge, m)
end

struct NestedModel{F <: Function,D <: Distribution} <: AbstractModel
    loglike::F
    priors::Vector{D}
end

struct NestedTransition{T} <: AbstractTransition
    active_points::Matrix{T}
    active_logl::Vector{T}
    samples::Vector{T}
    log_vol
    log_z
    h
end

function NestedTransition(model::NestedModel, nactive)
    ndim = length(model.priors)
    us = rand(ndim, nactive)
    ps = quantile.(hcat(model.priors), us)
    return NestedTransition(model, ps)
end

function NestedTransition(model::NestedModel, p::Matrix)
    # Get info from uniform space into prior space
    logls = [model.loglike(p[:, i]) for i in 1:size(p, 2)]

    # log prior volume
    logv = log(1 - exp(-1 / size(p, 2)))

    # log evidence
    logl_star, mindx = findmin(logls)

    # samples from least_likely
    s = p[:, mindx]

    return NestedTransition(p, logls, s, logv, -Inf, 0)
end

transition_type(model::NestedModel, spl::Nested) = NestedTransition

function step!(rng::AbstractRNG,
    model::NestedModel,
    spl::Nested,
    N::Integer;
    kwargs...)
    return NestedTransition(model, spl.nactive)
end

function propose(ell::Ellipsoid, model::NestedModel, logl_star)
    while true
        u = rand(ell)
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
    # We need to flush the remaining points in the ellipse at N-nactive
    if get(kwargs, :iteration, NaN) > N - spl.nactive
        i = spl.nactive - N + kwargs[:iteration]
        logl_star = prev.active_logl[i]
        log_wt = prev.log_vol + logl_star
        logz_new = log(exp(prev.log_z) + exp(log_wt))
        h = (exp(log_wt - logz_new) * logl_star + 
            exp(prev.log_z - logz_new) * (prev.h + prev.log_z) - logz_new)
        h = isnan(h) ? prev.h : h

        samples = prev.active_points[:, i]
        
        return NestedTransition(prev.active_points, prev.active_logl, samples, prev.log_vol, logz_new, h)
    end

    logl_star, mindx = findmin(prev.active_logl)
    log_wt = prev.log_vol + logl_star

    logz_new = log(exp(prev.log_z) + exp(log_wt))

    h = (exp(log_wt - logz_new) * logl_star + 
        exp(prev.log_z - logz_new) * (prev.h + prev.log_z) - logz_new)
    h = isnan(h) ? prev.h : h
    
    samples = prev.active_points[:, mindx]

    # Get points in unit space
    u = cdf.(hcat(model.priors), prev.active_points)

    # Get bounding ellipsoid
    enlarge_linear = spl.enlarge^(1 / size(prev.active_points, 1))
    ell = spl.method(u, enlarge_linear)
    p, logl = propose(ell, model, logl_star)

    log_vol = prev.log_vol - 1 / spl.nactive
    newp = prev.active_points
    newp[:, mindx] = p
    newlogl = prev.active_logl
    newlogl[mindx] = logl


    return NestedTransition(newp, newlogl, samples, log_vol, logz_new, h)
end

function bundle_samples(rng::AbstractRNG, 
    ℓ::AbstractModel, 
    s::Nested, 
    N::Integer, 
    ts::Vector{<:AbstractTransition}; 
    param_names = missing,
    kwargs...)
    vals = copy(reduce(hcat, [vcat(t.samples, t.log_z, t.h) for t in ts])')
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(first(vals)) - 2]
    end

    push!(param_names, "logz", "h")

    return Chains(vals, param_names, (internals = ["logz", "h"],))
end
