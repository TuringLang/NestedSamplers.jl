module Proposals

abstract type AbstractProposal end

struct Uniform <: AbstractProposal end

function (::Uniform)(u, prior_transform, loglike, args...; kwargs...)
    v = prior_transform(u)
    logl = loglike(v)
    return u, v, logl
end

end # module Proposals
