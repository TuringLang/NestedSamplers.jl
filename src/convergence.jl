# Convergence methods

"""
    decline_convergence(args...; decline_factor=6, kwargs...)

Stopping criterion: Number of consecutive declining log-evidence is greater than `iteration / decline_factor` or greater than `2nactive`
"""
function decline_convergence(rng::AbstractRNG,
    ::AbstractModel,
    sampler::Nested,
    transitions,
    iteration::Integer;
    progress = true,
    decline_factor = 6,
    kwargs...)
    # Don't accidentally short-circuit
    iteration > 2 || return false

    return sampler.ndecl > iteration / decline_factor || sampler.ndecl > 2sampler.nactive
end

"""
    dlogz_convergence(args...; dlogz=0.5, kwargs...)

Stopping criterion: estimated fraction evidence remaining below threshold.

The estimated fraction evidence remaining is given by the `maximum(active_loglike) - it/nactive` where `it` is the current iteration.
"""
function dlogz_convergence(rng::AbstractRNG,
    ::AbstractModel,
    sampler::Nested,
    transitions,
    iteration::Integer;
    progress = true,
    dlogz = 0.5,
    kwargs...)
    # Don't accidentally short-circuit
    iteration > 2 || return false

    logz_remain = maximum(sampler.active_logl) - (iteration - 1) / sampler.nactive
    dlogz_current = logaddexp(sampler.logz, logz_remain) - sampler.logz

    return dlogz_current < dlogz
end
