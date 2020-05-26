"""
    NestedSamplers.Proposals

This module contains the different algorithms for proposing new points within a bounding volume in unit space.

The available implementations are
* [`Proposals.Uniform`](@ref) - samples uniformly within the bounding volume
* [`Proposals.RWalk`](@ref) - random walks to a new point given an existing one
"""
module Proposals

using ..Bounds

using Random: AbstractRNG
using LinearAlgebra

export AbstractProposal

"""
    NestedSamplers.AbstractProposal <: Function

The abstract type for live point proposal algorithms.

# Interface

Each `AbstractProposal` must have this function, 
```julia
(::AbstractProposal)(::AbstractRNG, point, loglstar, bounds, loglikelihood, prior_transform)
```
which, given the input `point` with loglikelihood `loglstar` inside a `bounds`, returns a new point in unit space, prior space, and the loglikelihood.
"""
abstract type AbstractProposal <: Function end

function Base.show(io::IO, proposal::P) where P <: AbstractProposal
    base = nameof(P) |> string
    print(io, "$base(")
    fields = map(propertynames(proposal)) do name
        val = getproperty(proposal, name)
        "$(string(name))=$val"
    end
    join(io, fields, ", ")
    print(io, ")")
end

# ----------------------------------------

"""
    Proposals.Uniform()

Propose a new live point by uniformly sampling within the bounding volume.
"""
struct Uniform <: AbstractProposal end

function (::Uniform)(rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    loglike,
    prior_transform)
    while true
        u = rand(rng, bounds)
        all(p->0 < p < 1, u) || continue
        v = prior_transform(u)
        logl = loglike(v)
        logl ≥ logl_star && return u, v, logl
    end
end

"""
    Proposals.RWalk(;ratio=0.5, walks=25, scale=1)

Propose a new live point by random walking away from an existing live point.

`ratio` is the target acceptance ratio, `walks` is the minimum number of steps to take, and `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
Base.@kwdef mutable struct RWalk <: AbstractProposal
    ratio = 0.5
    walks = 25
    scale = 1
end

function (prop::RWalk)(rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    loglike,
    prior_transform;
    verbose=true,
    kwargs...)
    # setup
    n = length(point)
    axes = Bounds.axes(bounds)
    scale_init = prop.scale
    accept = reject = fail = nfail = ncall = 0
    local dr̂, dr, du, u_prop, logl_prop, u, v, logl

    while ncall < prop.walks || iszero(accept)
        # get proposed point
        while true
            # check scale factor to avoid over-shrinking
            prop.scale < 1e-5scale_init && error("Random walk sampling appears to be stuck.")
            # propose random direction in unit space
            dr̂ = randn(rng, n)
            dr̂ ./= LinearAlgebra.norm(dr̂)
            # scale based on dimensionality
            dr = @. dr̂ * rand(rng)^(1/n)
            # transform to proposal distribution
            du = axes * dr
            u_prop = @. point + prop.scale * du
            # inside unit-cube
            all(u -> 0 < u < 1, u_prop) && break
            fail += 1
            nfail += 1
            # check if stuck generating bad numbers
            if fail > 100prop.walks
                verbose && @warn "Random number generation appears extremely inefficient. Adjusting the scale-factor accordingly"
                fail = 0
                prop.scale *= exp(-1/n)
            end
        end
        # check proposed point
        v_prop = prior_transform(u_prop)
        logl_prop = loglike(v_prop)
        if logl_prop ≥ logl_star
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
        else
            reject += 1
        end
        ncall += 1
        
        # check if stuck generating bad points
        if ncall > 50prop.walks
            verbose && @warn "Random walk proposals appear to be extremely inefficient. Adjusting the scale-factor accordingly"
            prop.scale *= exp(-1/n)
            ncall = accept = reject = 0
        end
    end
    
    # update proposal scale
    ratio = accept / (accept + reject)
    norm = max(prop.ratio, 1 - prop.ratio) * n
    scale = prop.scale * exp((ratio - prop.ratio) / norm)
    prop.scale = min(scale, sqrt(n))

    return u, v, logl
end


end # module Proposals
