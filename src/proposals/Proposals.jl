"""
    NestedSamplers.Proposals

This module contains the different algorithms for proposing new points within a bounding volume in unit space.

The available implementations are
* [`Proposals.Rejection`](@ref) - samples uniformly within the bounding volume
* [`Proposals.RWalk`](@ref) - random walks to a new point given an existing one
* [`Proposals.RStagger`](@ref) - random staggering away to a new point given an existing one
* [`Proposals.Slice`](@ref) - slicing away to a new point given an existing one
* [`Proposals.RSlice`](@ref) - random slicing away to a new point given an existing one
"""
module Proposals

using ..NestedSamplers: prior_transform_and_loglikelihood
using ..Bounds

using Random
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

# Helper for checking unit-space bounds
unitcheck(us) = all(u -> 0 < u < 1, us)

"""
    Proposals.Rejection(;maxiter=100_000)

Propose a new live point by uniformly sampling within the bounding volume and rejecting samples that do not meet the likelihood constraints. This follows the original nested sampling algorithm proposed in Skilling (2004)[^1]

[^1]: John Skilling, 2004, AIP 735, 395  ["Nested Sampling"](https://aip.scitation.org/doi/abs/10.1063/1.1835238)

## Parameters
- `maxiter` is the maximum number of samples that can be rejected before giving up and throwing an error.
"""
Base.@kwdef struct Rejection <: AbstractProposal
    maxiter::Int = 100_000
end

@deprecate Uniform() Rejection()

function (prop::Rejection)(
    rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    model
)
    
    ncall = 0
    for _ in 1:prop.maxiter
        u = rand(rng, bounds)
        unitcheck(u) || continue
        v, logl = prior_transform_and_loglikelihood(model, u)
        ncall += 1
        logl ≥ logl_star && return u, v, logl, ncall
    end
    throw(ErrorException("Couldn't generate a proper point after $(prop.maxiter) attempts including $ncall likelihood calls. Bounds=$bounds, logl_star=$logl_star."))
end

Base.show(io::IO, p::Rejection) = print(io, "NestedSamplers.Proposals.Rejection")

"""
    Proposals.RWalk(;ratio=0.5, walks=25, scale=1)

Propose a new live point by random walking away from an existing live point. This follows the algorithm outlined in Skilling (2006).[^1]

[^1]: Skilling, 2006, Bayesian Anal. 1(4), ["Nested sampling for general Bayesian computation"](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full)

## Parameters
- `ratio` is the target acceptance ratio
- `walks` is the minimum number of steps to take
- `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
@with_kw mutable struct RWalk <: AbstractProposal
    ratio = 0.5
    walks = 25
    scale = 1.0

    @assert 1 / walks ≤ ratio ≤ 1 "Target acceptance ratio must be between 1/`walks` and 1"
    @assert walks > 1 "Number of steps must be greater than 1"
    @assert scale ≥ 0 "Proposal scale must be non-negative"
end

function (prop::RWalk)(
    rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    model;
    kwargs...
)
    # setup
    n = length(point)
    scale_init = prop.scale
    accept = reject = fail = nfail = nc = ncall = 0
    local du, u_prop, logl_prop, u, v, logl

    while nc < prop.walks || iszero(accept)
        # get proposed point
        while true
            # check scale factor to avoid over-shrinking
            prop.scale < 1e-5 * scale_init && error("Random walk sampling appears to be stuck.")
            # transform to proposal distribution
            du = randoffset(rng, bounds)
            u_prop = @. point + prop.scale * du
            # inside unit-cube
            unitcheck(u_prop) && break
            
            fail += 1
            nfail += 1
            # check if stuck generating bad numbers
            if fail > 100 * prop.walks
                @warn "Random number generation appears extremely inefficient. Adjusting the scale-factor accordingly"
                fail = 0
                prop.scale *= exp(-1/n)
            end
        end
        # check proposed point
        v_prop, logl_prop = prior_transform_and_loglikelihood(model, u_prop)
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
        if nc > 50 * prop.walks
            @warn "Random walk proposals appear to be extremely inefficient. Adjusting the scale-factor accordingly"
            prop.scale *= exp(-1/n)
            nc = accept = reject = 0
        end
    end
    
    # update proposal scale using acceptance ratio
    update_scale!(prop, accept, reject, n)

    return u, v, logl, ncall
end

# update proposal scale using target acceptance ratio
function update_scale!(prop, accept, reject, n)
    ratio = accept / (accept + reject)
    norm = max(prop.ratio, 1 - prop.ratio) * n
    scale = prop.scale * exp((ratio - prop.ratio) / norm)
    prop.scale = min(scale, sqrt(n))
    return prop
end

"""
    Proposals.RStagger(;ratio=0.5, walks=25, scale=1)

Propose a new live point by random staggering away from an existing live point. 
This differs from the random walk proposal in that the step size here is exponentially adjusted
to reach a target acceptance rate _during_ each proposal, in addition to _between_
proposals. This follows the algorithm outlined in Skilling (2006).[^1]

[^1]: Skilling, 2006, Bayesian Anal. 1(4), ["Nested sampling for general Bayesian computation"](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full)

## Parameters
- `ratio` is the target acceptance ratio
- `walks` is the minimum number of steps to take
- `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
@with_kw mutable struct RStagger <: AbstractProposal
    ratio = 0.5
    walks = 25
    scale = 1.0

    @assert 1 / walks ≤ ratio ≤ 1 "Target acceptance ratio must be between 1/`walks` and 1"
    @assert walks > 1 "Number of steps must be greater than 1"
    @assert scale ≥ 0 "Proposal scale must be non-negative"
end

function (prop::RStagger)(
    rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    model;
    kwargs...
)
    #setup
    n = length(point)
    scale_init = prop.scale
    accept = reject = fail = nfail = nc = ncall = 0
    stagger = 1
    local du, u_prop, logl_prop, u, v, logl
    
    while nc < prop.walks || iszero(accept)
        # get proposed point
        while true
            # check scale factor to avoid over-shrinking
            prop.scale < 1e-5 * scale_init && error("Random walk sampling appears to be stuck.")
            # transform to proposal distribution
            du = randoffset(rng, bounds)
            u_prop = @. point + prop.scale * stagger * du
            # inside unit-cube
            unitcheck(u_prop) && break
            
            fail += 1
            nfail += 1
            # check if stuck generating bad numbers
            if fail > 100 * prop.walks
                @warn "Random number generation appears extremely inefficient. Adjusting the scale-factor accordingly"
                fail = 0
                prop.scale *= exp(-1/n)
            end
        end 
        # check proposed point
        v_prop, logl_prop = prior_transform_and_loglikelihood(model, u_prop)
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
        
        # adjust _stagger_ to target an acceptance ratio of `prop.ratio`
        ratio = accept / (accept + reject)
        if ratio > prop.ratio
            stagger *= exp(1 / accept)
        elseif ratio < prop.ratio
            stagger /= exp(1 / reject)
        end
        
        # check if stuck generating bad points
        if nc > 50 * prop.walks
            @warn "Random walk proposals appear to be extremely inefficient. Adjusting the scale-factor accordingly"
            prop.scale *= exp(-1 / n)
            nc = accept = reject = 0
        end
    end
            
    # update proposal scale using acceptance ratio
    update_scale!(prop, accept, reject, n)
            
    return u, v, logl, ncall
end
      
"""
    Proposals.Slice(;slices=5, scale=1)

Propose a new live point by a series of random slices away from an existing live point.
This is a standard _Gibbs-like_ implementation where a single multivariate slice is a combination of `slices` univariate slices through each axis. This follows the algorithm outlined in Neal (2003).[^1]

[^1]: Neal, 2003, Ann. Statist. 31(3), ["Slice Sampling"](https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full)

## Parameters
- `slices` is the minimum number of slices
- `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
@with_kw mutable struct Slice <: AbstractProposal
    slices = 5
    scale = 1.0
    
    @assert slices ≥ 1 "Number of slices must be greater than or equal to 1"
    @assert scale ≥ 0 "Proposal scale must be non-negative"
end

function (prop::Slice)(
    rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    model;
    kwargs...
)
    # setup
    n = length(point)
    nc = nexpand = ncontract = 0
    local u, v, logl 
    
    # modifying axes and computing lengths
    axes = Bounds.axes(bounds)
    axes = prop.scale .* axes'
    # slice sampling loop
    for it in 1:prop.slices
        
        # shuffle axis update order
        idxs = shuffle!(rng, collect(Base.axes(axes, 1)))
        
        # slice sample along a random direction
        for idx in idxs
            
            # select axis
            axis = axes[idx, :]
            
            u, v, logl, nc, nexpand, ncontract = sample_slice(
                rng, axis, point, logl_star,
                model,
                nc, nexpand, ncontract
            )
        end # end of slice sample along a random direction             
    end # end of slice sampling loop    
    
    # update slice proposal scale based on the relative size of the slices compared to the initial guess
    prop.scale = prop.scale * nexpand / (2.0 * ncontract)
         
    return u, v, logl, nc
end   # end of function Slice  

"""
    Proposals.RSlice(;slices=5, scale=1)

Propose a new live point by a series of random slices away from an existing live point. This is a standard _random_ implementation where each slice is along a random direction based on the provided axes. This more closely matches the PolyChord implementation outlined in Handley et al. (2015a,b).[^1][^2]

[^1]: Handley, et al., 2015a, MNRAS 450(1), ["polychord: nested sampling for cosmology"](https://academic.oup.com/mnrasl/article/450/1/L61/986122)
[^2]: Handley, et al., 2015b, MNRAS 453(4), ["polychord: next-generation nested sampling"](https://academic.oup.com/mnras/article/453/4/4384/2593718)

## Parameters
- `slices` is the minimum number of slices
- `scale` is the proposal distribution scale, which will update _between_ proposals.
"""
@with_kw mutable struct RSlice <: AbstractProposal
    slices = 5
    scale = 1.0
    
    @assert slices ≥ 1 "Number of slices must be greater than or equal to 1"
    @assert scale ≥ 0 "Proposal scale must be non-negative"
end

function (prop::RSlice)(
    rng::AbstractRNG,
    point::AbstractVector,
    logl_star,
    bounds::AbstractBoundingSpace,
    model;
    kwargs...
)
    # setup
    n = length(point)
    nc = nexpand = ncontract = 0
    local u, v, logl
    # random slice sampling loop
    for it in 1:prop.slices
    
        # propose a direction on the unit n-sphere 
        drhat = randn(rng, n)
        drhat /= norm(drhat)
        
        # transform and scale into parameter space
        axis = prop.scale .* (Bounds.axes(bounds) * drhat)
        u, v, logl, nc, nexpand, ncontract = sample_slice(
            rng, axis, point, logl_star,
            model,
            nc, nexpand, ncontract
        )
    end # end of random slice sampling loop
    
    # update random slice proposal scale based on the relative size of the slices compared to the initial guess
    prop.scale = prop.scale * nexpand / (2.0 * ncontract)
    
    return u, v, logl, nc
end # end of function RSlice

# Method for slice sampling
function sample_slice(rng, axis, u, logl_star, model, nc, nexpand, ncontract)
    # define starting window
    r = rand(rng)  # initial scale/offset
    u_l = @. u - r * axis  # left bound
    if unitcheck(u_l)
        v_l, logl_l = prior_transform_and_loglikelihood(model, u_l)
    else
        logl_l = -Inf
    end
    nc += 1
    nexpand += 1 

    u_r = u_l .+ axis # right bound
    if unitcheck(u_r)
        v_r, logl_r = prior_transform_and_loglikelihood(model, u_r)
    else
        logl_r = -Inf
    end    
    nc += 1
    nexpand += 1 

    # stepping out left and right bounds
    while logl_l ≥ logl_star
        u_l .-= axis
        if unitcheck(u_l)
            v_l, logl_l = prior_transform_and_loglikelihood(model, u_l)
        else
            logl_l = -Inf
        end
        nc += 1
        nexpand += 1 
    end

    while logl_r ≥ logl_star
        u_r .+= axis
        if unitcheck(u_r)
            v_r, logl_r = prior_transform_and_loglikelihood(model, u_r)
        else
            logl_r = -Inf
        end
        nc += 1
        nexpand += 1 
    end

    # sample within limits. If the sample is not valid, shrink the limits until the `logl_star` bound is hit
    window_init = norm(u_r - u_l)  # initial window size
    while true 
        # define slice and window
        u_hat = u_r - u_l
        window = norm(u_hat)

        # check if the slice has shrunk to be ridiculously small
        window < 1e-5 * window_init && error("Slice sampling appears to be stuck.")

        # propose a new position
        r = rand(rng)
        u_prop = @. u_l + r * u_hat
        if unitcheck(u_prop)
            v_prop, logl_prop = prior_transform_and_loglikelihood(model, u_prop)
        else
            logl_prop = -Inf
        end
        nc += 1
        ncontract += 1

        # if success, then move to the new position
        if logl_prop ≥ logl_star
            return u_prop, v_prop, logl_prop, nc, nexpand, ncontract
        # if fail, then check if the new point is to the left/right of the original point along the proposal axis and update the bounds accordingly
        else
            s = dot(u_prop - u, u_hat) # check sign (+/-)
            if s < 0   # left
                u_l = u_prop
            elseif s > 0  # right
                u_r = u_prop
            else # if `s = 0` something has gone wrong
                error("Slice sampler has failed to find a valid point.")
            end
        end
    end # end of sample within limits while    
end    
    
end # module Proposals
