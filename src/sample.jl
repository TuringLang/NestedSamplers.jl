###
### Interface Implementations for running full sampling loops
###

using Printf

StatsBase.sample(rng::AbstractRNG, model::AbstractModel, sampler::Nested; kwargs...) =
    mcmcsample(rng, model, sampler, nested_isdone; progressname="Nested Sampling", chain_type=Chains, kwargs...)

StatsBase.sample(model::AbstractModel, sampler::Nested; kwargs...) =
    StatsBase.sample(GLOBAL_RNG, model, sampler; kwargs...)

function nested_isdone(rng, model, sampler, samples, state, i; progress=true, maxiter=Inf, maxcall=Inf, dlogz=0.5, maxlogl=Inf, kwargs...)
    # 1) iterations exceeds maxiter
    done_sampling = state.it ≥ maxiter
    # 2) number of loglike calls has been exceeded
    done_sampling |= state.ncall ≥ maxcall
    # 3) remaining fractional log-evidence below threshold
    logz_remain = maximum(state.logl) - state.it / sampler.nactive
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    done_sampling |= delta_logz ≤ dlogz
    # 4) last dead point loglikelihood exceeds threshold
    done_sampling |= state.logl_dead ≥ maxlogl
    # 5) number of effective samples
    # TODO

    if progress
        str = @sprintf "iter=%d\tncall=%d\tΔlogz=%.2g\tlogl=%.2g\tlogz=%.2g" i state.ncall delta_logz state.logl_dead state.logz
        print("\r\33[2K", str)
    end

    return done_sampling
end
