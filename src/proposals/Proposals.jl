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
using Distributions: Uniform

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
        unitcheck(u) || continue
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
proposals.

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
    Proposals.HSlice(;slices=5, scale=1)
Propose a new live point by "Hamiltonian" Slice Sampling using a series of random trajectories away from an existing live point.
Each trajectory is based on the provided axes and samples are determined by moving forwards/ backwards in time until the trajectory hits an edge 
and approximately reflecting off the boundaries.
After a series of reflections is established, a new live point is proposed by slice sampling across the entire path.
## Parameters
- `slices` is the minimum number of slices
- `scale` is the proposal distribution scale, which will update _between_ proposals
- `grad` is the gradient of the log-likelihood
- `max_move` is the limit for `ncall`
- `compute_jac` a true/false statement for whether the Jacobian is needed.
"""
@with_kw mutable struct HSlice <: AbstractProposal
    slices = 5
    scale = 1.0
    grad = nothing 
    max_move = 100
    compute_jac = false
    
    @assert slices ≥ 1 "Number of slices must be greater than or equal to 1"
    @assert scale ≥ 0 "Proposal scale must be non-negative"
    @assert max_move ≥ 1 "The limit for ncall must be greater than or equal to 1"
end

function (prop::HSlice)(rng::AbstractRNG,
                       point::AbstractVector,
                       logl_star,
                       bounds::AbstractBoundingSpace,
                       loglike,
                       prior_transform;
                       kwargs...)
    
    # setup
    n = length(point)
    jitter = 0.25 # 25% jitter
    nc = nmove = nreflect = ncontract = 0
    local   # incomplete
    
    # Hamiltonian slice sampling loop
    for it in 1:prop.slices
        # define the left, inner, and right nodes for a given chord
        # slice sampling will be done using these chords
        nodes_l = []
        nodes_m = []
        nodes_r = []
        
        # propose a direction on the unit n-sphere
        drhat = randn(rng, n)
        drhat /= norm(drhat)
        
        # transform and scale based on past tuning
        axes = Bounds.tran_axes(bounds)
        axis = dot(axes, drhat) * prop.scale * 0.01
        
        # creating starting window
        vel = axis    # current velocity
        u_l = @. u - Uniform(1.0 - jitter, 1.0 + jitter) * vel
        u_r = @. u + Uniform(1.0 - jitter, 1.0 + jitter) * vel
        append!(nodes_l, u_l)
        append!(nodes_m, u)
        append!(nodes_r, u_r)
        
        # progress right (i.e. forwards in time)
        reverse = false
        reflect = false
        u_r = u
        ncall = 0
        
        while ncall <= max_move
            
            # iterate until the edge of the distribution is bracketed
            append!(nodes_l, u_r)
            u_out = nothing
            u_in = []
            
            while true
            
                # step forward
                u_r += Uniform(1.0 - jitter, 1.0 + jitter) * vel
                
                # evaluate point
                if unitcheck(u_r)
                    v_r = prior_transform(u_r)
                    logl_r = loglike(v_r)
                    nc += 1
                    ncall += 1
                    nmove += 1
                else
                    logl_r = -Inf
                end    
                
                # check if the log-likelihood constraint is satisfied
                # (i.e. if in or out of bounds)
                
                if logl_r < logl_star
                    if reflect
                    # if out of bounds and just reflected, then reverse the direction and terminate immediately
                        reverse = true
                        pop!(nodes_l)   # remove since chord does not exist
                        break
                    else
                        # if already in bounds, then safe
                        u_out = u_r
                        logl_out = logl_r
                    end
                    # check if gradients can be computed assuming termination is with the current `u_out`
                    if isfinite(logl_out)
                        reverse = false
                    else
                        reverse = true
                    end
                else
                    reflect = false
                    append!(u_in, u_r)
                end 
                
                # check if the edge is bracketed
                if ## incomplete line 938
                    break
                end    
            end
            
            # define the rest of chord
            if ## incomplete
                
            end
            
            # check if turned around
            if reverse
                break
            end  
            
            # reflect off the boundary
            u_r = u_out
            logl_r = logl_out
            if ## incomplete
                # if the gradient is not provided, approximate it numerically using 2nd-order methods
                h = zeros(n)
                for i in 1:n
                    u_r_l = u_r
                    u_r_r = u_r
                    
                    # right side
                    u_r_r[i] += 1e-10
                    if unitcheck(u_r_r)
                        v_r_r = prior_transform(u_r_r)
                        logl_r_r = loglike(v_r_r)
                    else
                        logl_r_r = -Inf
                        reverse = true    # cannot compute gradient
                    end    
                    nc += 1
                    
                    # left side
                    u_r_l[i] += 1e-10
                    if unitcheck(u_r_l)
                        v_r_l = prior_transform(u_r_l)
                        logl_r_l = loglike(v_r_l)
                    else
                        logl_r_l = -Inf
                        reverse = true    # cannot compute gradient
                    end 
                    
                    if reverse
                        break    # give up because have to turn around
                    end    
                    nc += 1
                    
                    # compute dlnl/du
                    h[i] = (logl_r_r - logl_r_l) / 2e-10
                end
            else  
                # if the gradient is provided, evaluate it
                h = ## incomplete
                
                if compute_jac
                    jac = []
                    
                    # evaluate and apply Jacobian dv/du if gradient is defined as d(lnL)/dv instead of d(lnL)/du
                    for i in 1:n
                        u_r_l = u_r
                        u_r_r = u_r
                        
                        # right side
                        u_r_r[i] += 1e-10
                        if unitcheck(u_r_r)
                            v_r_r = prior_transform(u_r_r)
                        else
                            reverse = true    # cannot compute Jacobian
                            v_r_r = v_r    # assume no movement
                        end 
                        
                        # left side
                        u_r_l[i] -= 1e-10
                        if unitcheck(u_r_l)
                            v_r_l = prior_transform(u_r_l)
                        else  
                            reverse = true    # cannot compute Jacobian
                            v_r_r = v_r    # assume no movement
                        end  
                        
                        if reverse
                            break    # give up because have to turn around
                        end
                        
                        append!(jac, ((v_r_r - v_r_l) / 2e-10))
                    end 
                    
                    jac = jac
                    h = dot(jac, h)    # apply Jacobian
                end
                nc += 1
            end
            
            # compute specular reflection off boundary
            vel_ref = vel - 2 * h * dot(vel, h) / norm(h)^2
            dotprod = dot(vel_ref, vel)
            dotprod /= norm(vel_ref) * norm(vel)
            
            # check angle of reflection
            if dotprod < -0.99
                # the reflection angle is sufficiently small that it might as well be a reflection
                reverse = true
                break
            else
                # if reflection angle is sufficiently large, proceed as normal to the new position    
                vel = vel_ref
                u_out = nothing
                reflect = true
                nreflect += 1
            end    
        end
        
        # progress left (i.e. backwards in time)
        reverse = false
        reflect = false
        vel = axis    # current velocity
        u_l = u
        ncall = 0
        
        while ncall <= max_move
            
            # iterate until the edge of the distribution is bracketed
            # a doubling approach is used to try and locate the bounds faster
            append!(nodes_r, u_l)
            u_out = nothing
            u_in = []
            
            while true
                
                # step forward
                u_l += Uniform(1.0 - jitter, 1.0 + jitter) * vel
                
                # evaluate point
                if unitcheck(u_l)
                    v_l = prior_transform(u_l)
                    logl_l = loglike(v_l)
                    nc += 1
                    ncall += 1
                    nmove += 1
                else
                    logl_l = -Inf
                end  
                
                # check if the log-likelihood constraint are satisfied (i.e. in or out of bounds)
                if logl_l < logl_star
                    if reflect
                        # if out of bounds and just reflected, then reverse direction and terminate immediately
                        reverse = true
                        pop!(nodes_r)    # remove since chord does not exist
                        break
                    else
                        # if already in bounds, then safe
                        u_out = u_l
                        logl_out = logl_l
                    end 
                    
                    # check if gradients can be computed assuming there was termination with the current `u_out`
                    if isfinite(logl_out)
                        reverse = false
                    else
                        reverse = true
                    end  
                else 
                    reflect = false
                    append!(u_in, u_l)
                end 
                
                # check if the edge is bracketed
                if u_out ## incomplete
                    break
                end    
            end 
            
            # define the rest of chord
            if ## incomplete
                
            end  
            
            # check if turned around
            if reverse
                break
            end  
            
            # reflect off the boundary
            u_l = u_out
            logl_l = logl_out
            
            if grad ## incomplete
                
                # if the gradient is not provided, attempt to approximate it numerically using 2nd-order methods
                h = zeros(n)
                for i in 1:n
                    u_l_l = u_l
                    u_l_r = u_l
                    
                    # right side
                    u_l_r[i] += 1e-10
                    if unitcheck(u_l_r)
                        v_l_r = prior_transform(u_l_r)
                        logl_l_r = loglike(v_l_r)
                    else 
                        logl_l_r = -Inf
                        reverse = true    # cannot compute gradient
                    end
                    nc += 1
                    
                    # left side
                    u_l_l[i] -= 1e-10
                    if unitcheck(u_l_l)
                        v_l_l = prior_transform(u_l_l)
                        logl_l_l = loglike(v_l_l)
                    else 
                        logl_l_l = -Inf
                        reverse = true    # cannot compute gradient
                    end    
                    
                    if reverse
                        break    # give up because have to turn around
                    end
                    nc += 1
                    
                    # compute dlnl/du
                    h[i] = (logl_l_r - logl_l_l) / 2e-10 
                end    
            end
        else
            # if gradient is provided, evaluate it
            h = grad(v_l)
            if compute_jac
                jac = []
                
                # evaluate and apply Jacobian dv/du if gradient is defined as d(lnL)/dv instead of d(lnL)/du
                for i in 1:n
                    u_l_l = u_l
                    u_l_r = u_l
                    
                    # right side
                    u_l_r[i] += 1e-10
                    if unitcheck(u_l_r)
                        v_l_r = prior_transform(u_l_r)
                    else   
                        reverse = true    # cannot compute Jacobian
                        v_l_r = v_l    # assume no movement
                    end    
                    
                    # left side
                    u_l_l[i] -= 1e-10
                    if unitcheck(u_l_l)
                        v_l_l = prior_transform(u_l_l)
                    else    
                        reverse = true    # cannot compute Jacobian
                        v_l_r = v_l    # assume no movement
                    end 
                    
                    if reverse
                        break    # give up because have to turn around
                    end
                    
                    append!(jac, ((v_l_r - v_l_l) / 2e-10))
                end
                jac = jac
                h = dot(jac, h)    # apply Jacobian
            end    
            nc += 1
        end 
        
        # compute specular reflection off boundary
        vel_ref = vel - 2 * h * dot(vel, h) / norm(h)^2
        dotprod = dot(vel_ref, vel)
        dotprod /= norm(vel_ref) * norm(vel)
        
        # check angle of reflection
        if dotprod < -0.99
            # the reflection angle is sufficiently small that it might as well be a reflection
            reverse = true
            break
        else
            # if the reflection angle is sufficiently large, proceed as normal to the new position 
            vel = vel_ref
            u_out = nothing
            reflect = true
            nreflect += 1
        end 
    end
    
    # initialize lengths of cords
    if length(nodes_l) > 1
       
        # remove initial fallback chord
        popfirst!(nodes_l)
        popfirst!(nodes_m)
        popfirst!(nodes_r)
    end
    
    ## incomplete
    
    # slice sample from all chords simultaneously, this is equivalent to slice sampling in *time* along trajectory
    axlen_init = axlen
    
    while true
        
        # safety check
        if ## incomplete
            
        end 
        
        # select chord
        axprob = ## incomplete
        idx = ## incomplete
        
        # define chord
        u_l = nodes_l[idx]
        u_m = nodes_m[idx]
        u_r = nodes_r[idx]
        u_hat = u_r - u_l
        rprop = rand(rng)
        u_prop = @. u_l + rprop * u_hat    # scale from left
        if unitcheck(u_prop)
            v_prop = prior_transform(u_prop)
            logl_prop = loglike(v_prop)
        else
            logl_prop = -Inf
        end    
        nc += 1
        ncontract += 1
         
        # if succeed, move to the new position
        if logl_prop >= logl_star
            u = u_prop
            break
        end
        
        # if fail, check if the new point is to the left/right of the point interior to the bounds (`u_m`) and update the bounds accordingly
        else
            s = dot(u_prop - u_m, u_hat)    # check sign (+/-)
            if s < 0    # left
                nodes_l[idx] = u_prop
                axlen[idx] *= 1 - rprop
            elseif s > 0    # right
                nodes_r[idx] = u_prop
                axlen[idx] *= rprop
            else
                ## incomplete
            end
        ## check all the loops, & end statements
        ## also check where the placement of the update statement 
        ## also check placememt of return statement
    end  
    
    # update the Hamiltonian slice proposal scale based on the relative amount of time spent moving vs reflecting
    ## ncontract ... check this formula step
    ## check all formulas here, also check where to write prop.xyz and where not to write
    fmove = (1.0 * nmove) / (nmove + nreflect + ncontract + 2)
    norm = ## incomplete
    prop.scale *= ## incomplete
    
    return u_prop, v_prop, logl_prop, nc
    end    # end of function HSlice
    
    
end # module Proposals
