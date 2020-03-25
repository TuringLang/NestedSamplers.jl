#= This code is copy-pasted from working branch of AbstractMCMC:csp/infer
   it contains the logic for doing convergence-based sampling. This will be removed
   pending release of AbstractMCMC@0.5 =#

import StatsBase
using Random: GLOBAL_RNG
using ProgressMeter


"""
    sample([rng, ]model, sampler, N; kwargs...)

Return `N` samples from the MCMC `sampler` for the provided `model`.

If a callback function `f` with type signature 
```julia
f(rng::AbstractRNG, model::AbstractModel, sampler::AbstractSampler, N::Integer,
  iteration::Integer, transition; kwargs...)
```
may be provided as keyword argument `callback`. It is called after every sampling step.
"""
function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    arg...;
    kwargs...
)
    return sample(GLOBAL_RNG, model, sampler, arg...; kwargs...)
end

"""
    sample([rng::AbstractRNG, ]model::AbstractModel, s::AbstractSampler, is_done::Function; kwargs...)

`sample` will continuously draw samples without defining a maximum number of samples until
a convergence criteria defined by a user-defined function `is_done` returns `true`.

`is_done` is a function `f` that returns a `Bool`, with the signature

```julia
f(rng::AbstractRNG, model::AbstractModel, s::AbstractSampler, transitions::Vector, iteration::Int; kwargs...)
```

`is_done` should return `true` when sampling should end, and `false` otherwise.
"""
function StatsBase.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Nested,
    is_done = dlogz_convergence;
    chain_type::Type = Any,
    callback = (args...; kwargs...)->nothing,
    progress = true,
    kwargs...
)

    if progress
        pbar = ProgressThresh(get(kwargs, :dlogz, 0.5), "Sampling - dlogz: ")
    end

    sample_init!(rng, model, sampler, 1; kwargs...)

    # Obtain the initial transition.
    transition = step!(rng, model, sampler, 1; iteration = 1, kwargs...)

    # Run callback.
    callback(rng, model, sampler, 1, 1, transition; kwargs...)

    # Save the transition.
    transitions = [transition]

    # Step through the sampler until stopping.
    i = 2

    while !is_done(rng, model, sampler, transitions, i; pbar = pbar, kwargs...)
        # Obtain the next transition.
        transition = step!(rng, model, sampler, 1, transition; iteration = i, kwargs...)

        # Run callback.
        callback(rng, model, sampler, 1, i, transition; kwargs...)

        # Save the transition.
        push!(transitions, transition)

        # Increment iteration counter.
        i += 1
    end

    sample_end!(rng, model, sampler, 1, transitions; kwargs...)

    # Wrap the samples up.
    return bundle_samples(rng, model, sampler, i, transitions, chain_type; kwargs...)
end
