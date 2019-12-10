using Distributions
using Random
import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples, AbstractModel

abstract type NestedAlgorithm end
struct Single <: NestedAlgorithm end
struct Multi <: NestedAlgorithm end


"""
    Nested{<:NestedAlgorithm}(nactive, enlarge)

Nested Sampler

The two `NestedAlgorithm`s are `Single`, which uses a single bounding ellipsoid, and `Multi`, which finds an optimal clustering of ellipsoids.
"""
struct Nested{<:NestedAlgorithm} <: AbstractSampler 
    nactive::Integer
    enlarge::Float64
    starting_points::Matrix
end

Nested(nactive=100, enlarge=1.5) =  Nested{Single}(nactive, enlarge)

struct NestedModel{F<:Function, D<:Distribution} <: AbstractModel
    loglike::F
    priors::Vector{D}
end

struct NestedTransition{T} <: AbstractTransition
    active_points ::Matrix{T}
    active_logl   ::Vector{T}
    samples       ::Vector{T}
    log_vol       ::Float64
    log_wt        ::Float64
    log_z         ::Float64
    h             ::Float64
end

function NestedTransition(model::NestedModel, p::Matrix)
    # Get info from uniform space into prior space
    logls = [model.loglike(p[:, i]) for i in 1:size(p, 2)]

    # log prior volume
    logv = log(1 - exp(-1/size(p, 2)))

    # log evidence
    logl_star, mindx = findmin(logls)

    log_wt = logv + logl_star

    # samples from least_likely
    s = p[:, mindx]

    return NestedTransition(p, logls, s, logv, log_wt, -Inf, 0)
end

transition_type(model::NestedModel, spl::Nested) = NestedTransition

function step!(
    rng::AbstractRNG,
    model::NestedModel,
    spl::Nested{Single},
    N::Integer;
    kwargs...
)
    return NestedTransition(model, spl.starting_points)
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

function step!(
    rng::AbstractRNG,
    model::NestedModel,
    spl::Nested{Single},
    N::Integer,
    prev::NestedTransition;
    kwargs...
)
    logl_star, mindx = findmin(prev.active_logl)
    log_wt = prev.log_vol + logl_star

    logz_new = logaddexp(prev.logz, log_wt)
    h = (exp(log_wt - logz_new) * logl_star + 
        exp(prev.logz - logz_new) * (h + prev.logz) - logz_new)
    
    samples = prev.active_points[:, mindx]

    # Get points in unit space
    u = cdf.(hcat(model.priors), prev.active_points)

    # Get bounding ellipsoid
    enlarge_linear = spl.enlarge^(1/size(prev.active_points, 1))
    ell = fit(Ellipsoid, u, enlarge_linear)
    p, logl = propose(ell, model, logl_star)

    log_vol = prev.log_vol - 1/size(prev.active_points, 2)
    newp = prev.active_points
    newp[:, mindx] = p
    newlogl = prev.active_logl
    newlogl[mindx] = logl


    return NestedTransition(newp, newlogl, samples, log_vol, log_wt, logz_new, h)
end

function bundle_samples(

    rng::AbstractRNG, 
    ℓ::AbstractModel, 
    s::MetropolisHastings, 
    N::Integer, 
    ts::Vector{<:AbstractTransition}; 
    param_names=missing,
    kwargs...
)
    vals = copy(reduce(hcat, [vcat(t.samples, t.logz, t.h) for t in ts])')
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(first(vals)) - 2]
    end

    push!(param_names, "logz", "h")

    return Chains(vals, param_names, (internals=["logz", "h"],))
end
