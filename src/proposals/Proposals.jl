"""
    NestedSamplers.Proposals

This module contains the different algorithms for proposing new points within a bounding volume in unit space.

The available implementations are
* [`Proposals.Uniform`](@ref) - samples uniformly within the bounding volume
"""
module Proposals

using Random: AbstractRNG
using ..Bounds: AbstractBoundingSpace

export AbstractProposal

abstract type AbstractProposal end

function Base.show(io::IO, proposal::P) where P <: AbstractProposal
    base = nameof(P) |> string
    print(io, "$base(")
    fields = map(propertynames(proposal)) do name
        val = proposal.name
        "$(string(name))=$val"
    end
    join(io, fields, ", ")
    print(io, ")")
end

# ----------------------------------------

struct Uniform <: AbstractProposal end

function (::Uniform)(rng::AbstractRNG, bounds::AbstractBoundingSpace, loglike, prior_transform, logl_star, args...; kwargs...)
    while true
        u = rand(rng, bounds)
        all(p->0 < p < 1, u) || continue
        v = prior_transform(u)
        logl = loglike(v)
        logl ≥ logl_star && return u, v, logl
    end
end
end # module Proposals
