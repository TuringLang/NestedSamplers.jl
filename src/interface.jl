# Interface Implementations

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

# TODO


function transition()
    has_bounds::Bool
    active_us::Matrix{T}
    active_points::Matrix{T}
    active_logl::Vector{T}
    active_bound::B
    logz::Float64
    h::Float64
    log_vol::Float64
    ndecl::Int
    ncall::Int
    since_update::Int

    false,
    zeros(ndims, nactive),
    zeros(ndims, nactive),
    zeros(nactive),
    B,
    -1e300,
    0.0,
    log_vol,
    0,
    0,
    0
end

function AbstractMCMC.step(rng, model, sampler::Nested; kwargs...)
    # Initialize particles
    # us are in unit space, vs are in prior space
    us, vs, logl = init_particles(rng, model, sampler)

    # Initialize values for nested sampling loop.
    h = 0  # information, initially *0.*
    logz = -1e300  # ln(evidence), initially *0.*
    logavar = 0  # Var[ln(evidence)], initially *0.*
    logvol = 0  # initially contains the whole prior (volume=1.)
    loglstar = -1e300  # initial ln(likelihood)
    delta_logz = 1e300  # ln(ratio) of total/current evidence

    # Check if we should initialize a different bounding distribution
    # instead of using the unit cube.
    since_update = 0
    has_bounds = false

    # expected ln(vol) shrinkage
    logvol  -= sampler.dlnvol

    # Find live point with worst loglikelihood
    logl_dead, idx_dead = findmin(logl)
    draw = @view vs[idx_dead, :]
    logwt = logl_dead
    # Set our new weight using quadratic estimates (trapezoid rule).
    logdvol = log((1 - exp(logvol)) / 2) # ln(dvol)
    logwt = logl_dead + logdvol  # ln(wt)

    ###
    # TODO save this
    X⁺ = max(0, logvol)
    logdvol = X⁺ + log((exp(-X⁺) - exp(logvol - X⁺)) / 2)
    ###

    # Update evidence `logz` and information `h`.
    logz_new = logwt
    lzterm = exp(logl_dead - logz_new) * logl_dead
    h_new = (exp(logdvol) * lzterm +
             exp(logz - logz_new) * (h + logz) -
             logz_new)
    dh = h_new - h
    h = h_new
    logz = logz_new
    logzvar += 2 * dh * dlnvol
    loglstar = logl_dead

    sample = (u=@view(active_us[:, idx_dead]), v=draw, logwt=logwt, logl=logl_dead)
    state = (us=us, vs=vs, logl=logl, logl_dead=logl_dead, logwt=logwt,
             logz=logz, logzvar=logzvar, logvol=logvol,
             since_update=since_update, has_bounds=has_bounds, active_bound=nothing)

    return sample, state
end

function AbstractMCMC.step(rng, model, sampler, state; 
        dlogz=0.5, maxiter=Inf, maxcall=Inf, maxlogl=Inf, kwargs...)

    ## Step 1. Check stopping criterion
    # a) iterations exceeds maxiter
    done_sampling = state.it > maxiter
    # b) number of loglike calls has been exceeded
    done_sampling |= state.ncall > maxcall
    # c) remaining fractional log-evidence below threshold
    logz_remain = maximum(state.active_logl) + state.logvol
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    done_sampling |= delta_logz < dlogz
    # d) last dead point loglikelihood exceeds threshold
    done_sampling |= state.logl_dead > maxlogl

    ## Step 2. Update bounds
    pointvol = exp(state.logvol) / sampler.nactive
    # check if ready for first update
    if !state.has_bounds && state.ncall > sampler.min_ncall && state.it / state.ncall < sampler.min_eff
        @debug "First update: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = Bounds.scale!(Bounds.fit(B, state.us, pointvol = pointvol), sampler.enlarge)
        since_update = 0
        has_bounds = true
    # if accepted first update, is it time to update again?
    elseif iszero(state.since_update % state.update_interval)
        @debug "Updating bounds: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = Bounds.scale!(Bounds.fit(B, state.us, pointvol = pointvol), sampler.enlarge)
        since_update = 0
        has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update + 1
        has_bounds = state.has_bounds
    end

    ## Step 3. Replace least-likely active point
    # Find least likely point
    logl_dead, idx_dead = findmin(state.logl)
    u_dead = @view state.us[idx_dead, :]
    v_dead = @view state.vs[idx_dead, :]

    # update weight using trapezoidal rule
    logdvol = begin
        # weighted logaddexp
        X = state.logvol + sampler.dlnvol
        X⁺ = max(X, state.logvol)
        X⁺ + log((exp(X - X⁺) - exp(state.logvol - X⁺)) / 2)
    end
    logwt = logaddexp(logl_dead, state.logl_dead) + logdvol

    # sample a new live point using bounds and proposal
    if has_bounds
        point, bound = rand_live(rng, state.active_bound, state.us)
        u, v, logl, nc = s.proposal(rng, v_dead, logl_dead, bound, model.loglike, model.prior_transform)
    else
        point = rand(rng, T, s.ndims)
        bound = Bounds.NoBounds(T, s.ndims)
        proposal = Proposals.Uniform()
        u, v, logl, nc = proposal(rng, v_dead, logl_dead, bound, model.loglike, model.prior_transform)
    end
    state.us[:, idx_dead] .= u
    state.vs[:, idx_dead] .= v
    state.logl[:, idx_dead] .= logl

    ncall = state.ncall + nc
    since_update += nc

    # update evidence and information
    logz = logaddexp(state.logz, state.logwt)
    h = (exp(state.logwt - logz) * state.logl_dead +
           exp(state.logz - logz) * (state.h + state.logz) - logz)
    dh = h - state.h
    logzvar = state.logzvar + 2 * dh * sampler.dlnvol

    ## Part 4. prepare returns
    sample = (u=u_dead, v=v_dead, logwt=logwt, logl=logl_dead)
    state = (us=state.us, vs=state.vs, logl=state.logl, logl_dead=logl_dead, logwt=logwt,
             logz=logz, logzvar=logzvar, logvol=logvol,
             since_update=since_update, has_bounds=has_bounds, active_bound=active_bound)

    return sample, state
end

## Helpers

init_particles(rng, nactive, ndims, prior, loglike) =
    init_particles(rng, Float64, nactive, ndims, prior, loglike)

init_particles(rng, model, sampler) =
    init_particles(rng, sampler.nactive, sampler.ndims, model.prior_transform, model.loglike)

# loop and fill arrays, checking validity of points
# will retry 100 times before erroring
function init_particles(rng, T, nactive, ndims, prior, loglike)
    local us, vs, logl
    ntries = 0
    while true
        us = rand(rng, T, nactive, ndims)
        vs = mapslices(prior, us, dims=2)
        logl = mapslices(loglike, vs, dims=2)
        any(isfinite, logl) && break
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood methods.")
    end

    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    return us, vs, logl
end