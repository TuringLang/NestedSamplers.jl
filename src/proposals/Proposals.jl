"""
    NestedSamplers.Proposals

This module contains the different algorithms for proposing new points within a bounding volume in unit space.

The available implementations are
* [`Proposals.Uniform`](@ref) - samples uniformly within the bounding volume
* [`Proposals.RWalk`](@ref) - random walks to a new point given an existing one
* [`Proposals.RStagger`](@ref) - random staggering away to a new point given an existing one
"""
module Proposals

using ..Bounds

using Random: AbstractRNG
using LinearAlgebra
using Parameters

export AbstractProposal

"""
    NestedSamplers.AbstractProposal

The abstract type for live point proposal algorithms.

# Interface

Each `AbstractProposal` must have this function, 
```julia
(::AbstractProposal)(::AbstractRNG, point, loglstar, bounds, loglikelihood, prior_transform)
```
which, given the input `point` with loglikelihood `loglstar` inside a `bounds`, returns a new point in unit space, prior space, the loglikelihood, and the number of function calls.
"""
abstract type AbstractProposal end

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
    
    ncall = 0
    while true
        u = rand(rng, bounds)
        all(p->0 < p < 1, u) || continue
        v = prior_transform(u)
        logl = loglike(v)
        ncall += 1
        logl ≥ logl_star && return u, v, logl, ncall
    end
end

Base.show(io::IO, p::Uniform) = print(io, "NestedSamplers.Proposals.Uniform")

"""
    Proposals.RWalk(;ratio=0.5, walks=25, scale=1)

Propose a new live point by random walking away from an existing live point.

`ratio` is the target acceptance ratio, `walks` is the minimum number of steps to take, and `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
@with_kw mutable struct RWalk <: AbstractProposal
    ratio = 0.5
    walks = 25
    scale = 1.0
end

function (prop::RWalk)(rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    loglike,
    prior_transform;
    kwargs...)
    # setup
    n = length(point)
    scale_init = prop.scale
    accept = reject = fail = nfail = nc = ncall = 0
    local dr̂, dr, du, u_prop, logl_prop, u, v, logl

    while nc < prop.walks || iszero(accept)
        # get proposed point
        while true
            # check scale factor to avoid over-shrinking
            prop.scale < 1e-5scale_init && error("Random walk sampling appears to be stuck.")
            # transform to proposal distribution
            du = randoffset(rng, bounds)
            u_prop = @. point + prop.scale * du
            # inside unit-cube
            all(u -> 0 < u < 1, u_prop) && break
            
            fail += 1
            nfail += 1
            # check if stuck generating bad numbers
            if fail > 100prop.walks
                @warn "Random number generation appears extremely inefficient. Adjusting the scale-factor accordingly"
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
        nc += 1
        ncall += 1
        
        # check if stuck generating bad points
        if ncall > 50prop.walks
            @warn "Random walk proposals appear to be extremely inefficient. Adjusting the scale-factor accordingly"
            prop.scale *= exp(-1/n)
            nc = accept = reject = 0
        end
    end
    
    # update proposal scale
    ratio = accept / (accept + reject)
    norm = max(prop.ratio, 1 - prop.ratio) * n
    scale = prop.scale * exp((ratio - prop.ratio) / norm)
    prop.scale = min(scale, sqrt(n))

    return u, v, logl, ncall
end

"""
    Proposals.RStagger(;ratio=0.5, walks=25, scale=1)  (note-to-self: check if more kwargs... go in here)

Propose a new live point by random staggering away from an existing live point. 
This differs from the random walk proposal in that the step size here is exponentially adjusted
to reach a target acceptane rate _during_ each proposal, in addition to _between_
proposals.

`ratio` is the target acceptance ratio, `walks` is the minimum number of steps to take, and `scale` is the proposal distribution scale, which will update (_during_ and?) _between_ proposals.
"""

@with_kw mutable struct RStagger <: AbstractProposal
    ratio = 0.5
    walks = 25
    scale = 1.0
end

function (prop::RStagger)(rng::AbstractRNG,
        point::AbstractVector,
        logl_star,
        bounds::AbstractBoundingSpace,
        loglike,
        prior_transform;
        kwargs...)
        #setup
        n = length(point)
        scale_init = prop.scale
        accept = reject = fail = nfail = nc = ncall = 0
        stagger = 1
        local drhat, dr, du, u_prop, logl_prop (u, v, logl not to define here?)
    
        while nc < prop.walks || iszero(accept)
            # get proposed point
            while true:
                # check scale factor to avoid over-shrinking
                prop.scale < 1e-5scale_init && error("Random walk sampling appears to be stuck.")
                
                # transform to proposal distribution
                du = randoffset(rng, bounds)
                u_prop = @. point + prop.scale*stagger*du
                # inside unit-cube
                all(u -> 0 < u < 1, u_prop) && break
            
                fail += 1
                nfail += 1
            
                # check if stuck generating bad numbers
                if fail > 100prop.walks
                    @warn "Random number generation appears extremely inefficient. Adjusting the scale-factor accordingly"
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
                logl = logl1_prop
                accept += 1
            else
                reject += 1
            end
            nc += 1
            ncall += 1
        
            # adjust _stagger_ to target an acceptance ratio of `ratio`
            
        
        
        
            # check if stuck generating bad points
            if nc > 50prop.walks
                @warn "Random walk proposals appear to be extremely inefficient. Adjusting the scale-factor accordingly"
                prop.scale *= exp(-1/n)
                nc = accept = reject = 0
            end
        end
  
end # module Proposals
