function decline_covergence(rng::AbstractRNG,
    model::AbstractModel,
    s::Nested,
    ts::Vector{<:AbstractTransition},
    iteration::Integer;
    decline_factor = 1,
    kwargs...)
    return s.ndecl > decline_factor * iteration
end

function dlogz_convergence(rng::AbstractRNG,
    model::AbstractModel,
    s::Nested,
    ts::Vector{<:AbstractTransition},
    iteration::Integer;
    pbar,
    dlogz = 0.5,
    kwargs...)
     #= Stopping criterion: estimated fraction evidence remaining 
    below threshold =#
    logz_remain = maximum(s.active_logl) - (iteration - 1) / s.nactive
    dlogz_current = logaddexp(s.logz, logz_remain) - s.logz
    ProgressMeter.update!(pbar, dlogz_current)
    return dlogz_current < dlogz
end
