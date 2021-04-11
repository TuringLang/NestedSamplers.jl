
function step(rng, model, sampler::Nested; kwargs...)
    # Initialize particles
    # us are in unit space, vs are in prior space
    us, vs, logl = init_particles(rng, model, sampler)

    # Find least likely point
    logl_dead, idx_dead = findmin(logl)
    u_dead = @view us[:, idx_dead]
    v_dead = @view vs[:, idx_dead]

    # update weight using trapezoidal rule
    logvol = -sampler.dlnvol
    logdvol = log((1 - exp(logvol)) / 2)
    logwt = logl_dead + logdvol

    # sample a new live point without bounds
    point = rand(rng, eltype(us), sampler.ndims)
    bound = Bounds.fit(Bounds.NoBounds, us)
    proposal = Proposals.Uniform()
    u, v, ll, nc = proposal(rng, v_dead, logl_dead, bound, model.loglike, model.prior_transform)

    us[:, idx_dead] .= u
    vs[:, idx_dead] .= v
    logl[idx_dead] = ll

    ncall = since_update = nc

    # update evidence and information
    logz = logwt
    h = logz
    logzvar = 2 * h * sampler.dlnvol

    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = 1, ncall = ncall, us = us, vs = vs, logl = logl, logl_dead = logl_dead, logwt = logwt,
             logz = logz, logzvar = logzvar, h = h, logvol = logvol,
             since_update = since_update, has_bounds = false, active_bound = nothing)

    return sample, state
end

function step(rng, model, sampler, state; kwargs...)
    ## Update bounds
    pointvol = exp(state.logvol) / sampler.nactive
    # check if ready for first update
    if !state.has_bounds && state.ncall > sampler.min_ncall && state.it / state.ncall < sampler.min_eff
        @debug "First update: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = Bounds.scale!(Bounds.fit(sampler.bounds, state.us, pointvol=pointvol), sampler.enlarge)
        since_update = 0
        has_bounds = true
    # if accepted first update, is it time to update again?
    elseif iszero(state.since_update % sampler.update_interval)
        @debug "Updating bounds: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = Bounds.scale!(Bounds.fit(sampler.bounds, state.us, pointvol=pointvol), sampler.enlarge)
        since_update = 0
        has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update + 1
        has_bounds = state.has_bounds
    end

    ## Replace least-likely active point
    # Find least likely point
    logl_dead, idx_dead = findmin(state.logl)
    u_dead = @view state.us[:, idx_dead]
    v_dead = @view state.vs[:, idx_dead]

    # update weight using trapezoidal rule
    logvol = state.logvol - sampler.dlnvol
    logdvol = begin
        # weighted logaddexp
        X⁺ = max(state.logvol, logvol)
        X⁺ + log((exp(state.logvol - X⁺) - exp(logvol - X⁺)) / 2)
    end
    logwt = logaddexp(logl_dead, state.logl_dead) + logdvol

    # sample a new live point using bounds and proposal
    if has_bounds
        point, bound = rand_live(rng, active_bound, state.us)
        u, v, logl, nc = sampler.proposal(rng, point, logl_dead, bound, model.loglike, model.prior_transform)
    else
        point = rand(rng, eltype(state.us), sampler.ndims)
        bound = Bounds.fit(Bounds.NoBounds, state.us)
        proposal = Proposals.Uniform()
        u, v, logl, nc = proposal(rng, point, logl_dead, bound, model.loglike, model.prior_transform)
    end
    state.us[:, idx_dead] .= u
    state.vs[:, idx_dead] .= v
    state.logl[idx_dead] = logl

    ncall = state.ncall + nc
    since_update += nc

    # update evidence and information
    logz = logaddexp(state.logz, state.logwt)
    h = (exp(state.logwt - logz) * state.logl_dead +
         exp(state.logz - logz) * (state.h + state.logz) - logz)
    dh = h - state.h
    logzvar = state.logzvar + 2 * dh * sampler.dlnvol

    ## prepare returns
    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = state.it + 1, ncall = ncall, us = state.us, vs = state.vs, logl = state.logl, logl_dead = logl_dead, logwt = logwt,
             logz = logz, logzvar = logzvar, h = h, logvol = logvol,
             since_update = since_update, has_bounds = has_bounds, active_bound = active_bound)

    return sample, state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Chains};
        add_live=true,
        param_names=missing,
        check_wsum=true,
        kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end
    vals = mapreduce(t -> hcat(t.v..., t.logwt), vcat, samples)
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - state.logz)
    wsum = sum(vals[:, end, 1])
    @. vals[:, end, 1] /= wsum

    if check_wsum
        err = !iszero(state.logzvar) ? 3 * sqrt(state.logzvar) : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence=state.logz), state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Array};
        add_live=true,
        check_wsum=true,
        kwargs...)
    
    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end

    wsum = sum(s -> exp(s.logwt - state.logz), samples)

    if check_wsum
        err = !iszero(state.logzvar) ? 3 * sqrt(state.logzvar) : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    vals = mapreduce(t -> hcat(t.v..., t.logwt / wsum), vcat, samples)

    return vals, state
end


function bundle_samples(samples,
    model::AbstractModel,
    sampler::Nested,
    state,
    ::Type{Vector};
    param_names=missing,
    add_live=true,
    check_wsum=true,
    kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end

    wsum = sum(s -> exp(s.logwt - state.logz), samples)

    if check_wsum
        err = !iszero(state.logzvar) ? 3 * sqrt(state.logzvar) : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = [Symbol("Parameter $i") for i in 1:sampler.ndims]
    end
    push!(param_names, :weights)

    vals = map(samples) do sample
        data = (sample.v..., sample.logwt / wsum)
        NamedTuple(zip(param_names, data))
    end

    return vals, state
end

## Helpers

init_particles(rng, ndims, nactive, prior, loglike) =
    init_particles(rng, Float64, ndims, nactive, prior, loglike)

init_particles(rng, model, sampler) =
    init_particles(rng, sampler.ndims, sampler.nactive, model.prior_transform, model.loglike)

# loop and fill arrays, checking validity of points
# will retry 100 times before erroring
function init_particles(rng, T, ndims, nactive, prior, loglike)
    us = rand(rng, T, ndims, nactive)
    vs = mapslices(prior, us, dims=1)
    logl = dropdims(mapslices(loglike, vs, dims=1), dims=1)
        ntries = 1
    while true
        any(isfinite, logl) && break
        us .= rand(rng, T, ndims, nactive)
        vs .= mapslices(prior, us, dims=1)
        logl .= mapslices(loglike, vs, dims=1)
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood methods.")
    end

    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    return us, vs, logl
end


# add remaining live points to `samples`
function add_live_points(samples, model, sampler, state)
    N = length(samples)

    prev_logvol = state.logvol
    prev_logz = state.logz
    prev_logzvar = state.logzvar
    prev_h = state.h
    prev_ll = state.logl_dead

    sorted_idxs = sortperm(state.logl)

    dlnvol = -N / sampler.nactive

    local logwt

    @inbounds for (i, idx) in enumerate(sorted_idxs)
        # get new point
        u = @view state.us[:, idx]
        v = @view state.vs[:, idx]
        ll = state.logl[idx]

        logvol = prev_logvol + log(1 - i / (sampler.nactive + 1))
        logdvol = begin
            # weighted logaddexp
            X⁺ = max(prev_logvol, logvol)
            X⁺ + log((exp(prev_logvol - X⁺) - exp(logvol - X⁺))) - log(2)
        end
        logwt = logaddexp(ll, prev_ll) + logdvol
        
        # update sampler
        logz = logaddexp(prev_logz, logwt)
        lzterm = (exp(prev_ll - logz) * prev_ll +
                  exp(ll - logz) * ll)
        h = (exp(logdvol) * lzterm +
             exp(prev_logz - logz) * (prev_h + prev_logz) - logz)
        dh = h - prev_h
        logzvar = prev_logzvar + 2 * dh * dlnvol

        prev_logz = logz
        prev_logzvar = logzvar
        prev_h = h
        prev_ll = ll

        sample = (u = u, v = v, logwt = logwt, logl = ll)
        save!!(samples, sample, N + i, model, sampler)

        i += 1
    end

    state = (it = N + sampler.nactive, us = state.us, vs = state.vs, logl = state.logl, logl_dead = prev_ll, logwt = logwt,
            logz = prev_logz, logzvar = prev_logzvar, logvol = prev_logvol,
            since_update = state.since_update, has_bounds = state.has_bounds, active_bound = state.active_bound)
    return samples, state
end
