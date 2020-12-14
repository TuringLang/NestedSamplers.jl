# Interface Implementations for running full sampling loops

StatsBase.sample(rng::AbstractRNG, model, sampler::Nested; kwargs...) =
    AbstractMCMC.mcmcsample(rng, model, sampler; kwargs...)

StatsBase.sample(model, sampler::Nested; kwargs...) =
    sample(Random.GLOBAL_RNG, model, sampler; kwargs...)

function isdone(state; dlogz=0.5, maxiter=Inf, maxcall=Inf, maxlogl=Inf, kwargs...)
    # 1) iterations exceeds maxiter
    done_sampling = state.it > maxiter
    # 2) number of loglike calls has been exceeded
    done_sampling |= state.ncall > maxcall
    # 3) remaining fractional log-evidence below threshold
    logz_remain = maximum(state.active_logl) + state.logvol
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    done_sampling |= delta_logz < dlogz
    # 4) last dead point loglikelihood exceeds threshold
    done_sampling |= state.logl_dead > maxlogl
    # 5) number of effective samples
    # TODO
    return done_sampling
end


function AbstractMCMC.mcmcsample(
    rng,
    model,
    sampler::Nested;
    kwargs...
)
    @ifwithprogresslogger progress name=progressname begin
        # Obtain the initial sample and state.
        sample, state = step(rng, model, sampler; kwargs...)

        # Discard initial samples.
        for _ in 2:discard_initial
            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, 1)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler; kwargs...)

        # Step through the sampler until stopping.
        i = 2

        while !isdone(state; kwargs...)
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = step(rng, model, sampler, state; kwargs...)
            end
            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler; kwargs...)
            # Increment iteration counter.
            i += 1
        end
    end

    # Wrap the samples up.
    return bundle_samples(samples, model, sampler, state, chain_type; kwargs...)
end
