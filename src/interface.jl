# Interface Implementations

function sample_init!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    ::Integer;
    debug::Bool = false,
    kwargs...) where {E <: AbstractEllipsoid}

    debug && @info "Initializing sampler"
    s.nactive < 2length(model.priors) && @warn "Using fewer than 2ndim active points is discouraged"

    # samples in unit space
    us = rand(rng, length(model.priors), s.nactive)

    # samples and loglikes in prior space
    s.active_points = quantile.(hcat(model.priors), us)
    s.active_logl = @inbounds [model.loglike(s.active_points[:, i]) for i in eachindex(s.active_logl)]
    
    any(isinf.(s.active_logl)) && @warn "Infinite log-likelihood found initializing sampler. This will cause failure to accurately calculate the information, h. Double check your log-likelihood function is numerically stable"

    # get bounding ellipsoid
    s.active_ell = scale!(fit(E, us, pointvol = 1 / s.nactive), s.enlarge)

    return nothing
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested,
    ::Integer;
    kwargs...)
    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]
    log_wt = s.log_vol + logL

    # update sampler
    logz = logaddexp(s.logz, log_wt)
    s.h = (exp(log_wt - logz) * logL +
           exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    return NestedTransition(draw, logL, log_wt)
end

function step!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{E},
    ::Integer,
    prev::NestedTransition;
    iteration,
    debug::Bool = false,
    kwargs...) where {E <: AbstractEllipsoid}
    
    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]
    log_wt = s.log_vol + logL

    # update evidence and information
    logz = logaddexp(s.logz, prev.log_wt)
    s.h = (exp(prev.log_wt - logz) * prev.logL +
           exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    # Get bounding ellipsoid (only every update_interval)
    if iszero(iteration % s.update_interval)
        # Get points in unit space
        u = cdf.(hcat(model.priors), s.active_points)

        # fit ellipsoid
        pointvol = exp(-(iteration - 1) / s.nactive) / s.nactive
        s.active_ell = scale!(fit(E, u, pointvol = pointvol), s.enlarge)
    end

    # Get new point and log like
    p, logl = propose(rng, s.active_ell, model, logL)
    @inbounds s.active_points[:, idx] = p
    @inbounds s.active_logl[idx] = logl
    s.ndecl = log_wt < prev.log_wt ? s.ndecl + 1 : 0

    # Shrink interval
    s.log_vol -=  1 / s.nactive

    return NestedTransition(draw, logL, log_wt)
end

function sample_end!(rng::AbstractRNG,
    ℓ::AbstractModel,
    s::Nested,
    ::Integer,
    transitions;
    debug::Bool = false,
    kwargs...)
    # Pop remaining points in ellipsoid
    N = length(transitions)
    s.log_vol = -N / s.nactive - log(s.nactive)
    @inbounds for i in eachindex(s.active_logl)
        # get new point
        draw = s.active_points[:, i]
        logL = s.active_logl[i]
        log_wt = s.log_vol + logL

        # update sampler
        logz = logaddexp(s.logz, log_wt)
        s.h = (exp(log_wt - logz) * logL +
               exp(s.logz - logz) * (s.h + s.logz) - logz)
        s.logz = logz

        prev = NestedTransition(draw, logL, log_wt)
        push!(transitions, prev)
    end

    # h should always be non-negative. Numerical error can arise from pathological corner cases
    if s.h < 0
        s.h ≉ 0 && @warn "Negative h encountered h=$(s.h). This is likely a bug"
        s.h = zero(s.h)
    end

    return nothing
end

function bundle_samples(rng::AbstractRNG,
    ::AbstractModel,
    s::Nested,
    N::Integer,
    transitions,
    Chains;
    param_names = missing,
    check_wsum = true,
    kwargs...)

    vals = copy(mapreduce(t->vcat(t.draw, t.log_wt), hcat, transitions)')
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - s.logz)
    wsum = sum(vals[:, end, 1])
    @. vals[:, end, 1] /= wsum

    if check_wsum
        err = !iszero(s.h) ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence = exp(s.logz))
end

function bundle_samples(rng::AbstractRNG,
    ::AbstractModel,
    s::Nested,
    N::Integer,
    transitions,
    A::Type{<:AbstractArray};
    check_wsum = true,
    kwargs...)

    vals = convert(A, mapreduce(t->t.draw, hcat, transitions)')
    
    if check_wsum
        # get weights
        wsum = mapreduce(t->exp(t.log_wt - s.logz), +, transitions)

        # check with h
        err = s.h ≠ 0 ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    return vals
end

"""
Propose a new point in the given `AbstractEllipsoid` that is guaranteed to have log-likelihood greater than or equal to `logl_star`
"""
function propose(rng::AbstractRNG, ell::AbstractEllipsoid, model::NestedModel, logl_star)
    while true
        u = rand(rng, ell)
        all(0 .< u .< 1) || continue
        v = quantile.(model.priors, u)
        logl = model.loglike(v)
        logl ≥ logl_star && return v, logl
    end
end

# Use to set default convergence metric
function StatsBase.sample(
    rng::AbstractRNG,
    model::NestedModel,
    sampler::Nested;
    kwargs...
)
    mcmcsample(rng, model, sampler, dlogz_convergence; kwargs...)
end

# Use to set default convergence metric
function StatsBase.sample(
    model::NestedModel,
    sampler::Nested;
    kwargs...
)
    sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end