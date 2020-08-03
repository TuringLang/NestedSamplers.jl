# Interface Implementations

function AbstractMCMC.step(rng::AbstractRNG,
    model,
    sampler::Nested;
    kwargs...)

    ## Sample Init
    debug && @info "Initializing sampler"
    local us, vs, logl
    ntries = 0
    while true
        us = rand(rng, s.ndims, s.nactive)
        vs = mapslices(model.prior_transform, us, dims=1)
        logl = mapslices(model.loglike, vs, dims=1)
        any(isfinite, logl) && break
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood method.")
    end
    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    # samples in unit space
    s.active_us .= us
    s.active_points .= vs
    s.active_logl .= logl[1, :]

    ## Step!
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

function AbstractMCMC.step(rng::AbstractRNG,
    model,
    sampler::Nested,
    state;
    kwargs...)
end


function sample_init!(rng::AbstractRNG,
    model::NestedModel,
    s::Nested{T,B},
    ::Integer;
    debug::Bool = false,
    kwargs...) where {T,B}

    debug && @info "Initializing sampler"
    local us, vs, logl
    ntries = 0
    while true
        us = rand(rng, s.ndims, s.nactive)
        vs = mapslices(model.prior_transform, us, dims=1)
        logl = mapslices(model.loglike, vs, dims=1)
        any(isfinite, logl) && break
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood method.")
    end
    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    # samples in unit space
    s.active_us .= us
    s.active_points .= vs
    s.active_logl .= logl[1, :]

    return nothing
end

function step!(::AbstractRNG,
    ::AbstractModel,
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
    s::Nested{T,B},
    ::Integer,
    prev::NestedTransition;
    iteration,
    debug::Bool = false,
    kwargs...) where {T,B}

    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]
    log_wt = s.log_vol + logL

    # update evidence and information
    logz = logaddexp(s.logz, prev.log_wt)
    s.h = (exp(prev.log_wt - logz) * prev.logL +
           exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    # check if ready for first update
    if !s.has_bounds && s.ncall > s.min_ncall && iteration / s.ncall < s.min_eff
        debug && @info "First update: it=$iteration, ncall=$(s.ncall), eff=$(iteration / s.ncall)"
        s.has_bounds = true
        pointvol = exp(s.log_vol) / s.nactive
        s.active_bound = Bounds.scale!(Bounds.fit(B, s.active_us, pointvol = pointvol), s.enlarge)
        s.since_update = 0
    # if accepted first update, is it time to update again?
    elseif iszero(s.since_update % s.update_interval)
        debug && @info "Updating bounds: it=$iteration, ncall=$(s.ncall), eff=$(iteration / s.ncall)"
        pointvol = exp(s.log_vol) / s.nactive
        s.active_bound = Bounds.scale!(Bounds.fit(B, s.active_us, pointvol = pointvol), s.enlarge)
        s.since_update = 0
    end
    
    # Get a live point to use for evolving with proposal
    if s.has_bounds
        point, bound = rand_live(rng, s.active_bound, s.active_us)
        u, v, logl, ncall = s.proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    else
        point = rand(rng, T, s.ndims)
        bound = Bounds.NoBounds(T, s.ndims)
        proposal = Proposals.Uniform()
        u, v, logl, ncall = proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    end

    # Get new point and log like
    s.active_us[:, idx] = u
    s.active_points[:, idx] = v
    s.active_logl[idx] = logl
    s.ndecl = log_wt < prev.log_wt ? s.ndecl + 1 : 0
    s.ncall += ncall
    s.since_update += 1

    # Shrink interval
    s.log_vol -=  1 / s.nactive

    return NestedTransition(draw, logL, log_wt)
end

function sample_end!(::AbstractRNG,
    ::AbstractModel,
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

function bundle_samples(::AbstractRNG,
    ::AbstractModel,
    s::Nested,
    ::Integer,
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

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence = s.logz)
end

function bundle_samples(::AbstractRNG,
    ::AbstractModel,
    s::Nested,
    ::Integer,
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
    StatsBase.sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end
