using NestedSamplers
using Test
using Random
using StableRNGs
using LinearAlgebra
using IterTools

rng = StableRNG(8425)

@testset "Bounds" begin include("bounds/bounds.jl") end
@testset "Proposals" begin include("proposals/proposals.jl") end
@testset "Sampler" begin include("sampler.jl") end

function integrate_on_grid(f, ranges, density)
    rs = []
    for r in ranges
        step = (r[2] - r[1]) / density
        rmin = r[1] + step / 2
        rmax = r[2] - step / 2
        push!(rs, range(rmin, rmax, length = density))
    end

    logsum = -1e300
    for v in Iterators.product(rs...)
        logsum = log(exp(logsum) + f(v))
    end
    logsum -= length(ranges) * log(density)

    return logsum
end

function integrate_on_grid(f, ranges)
    density = 100
    logsum_old = -Inf
    while true
        logsum = integrate_on_grid(f, ranges, density)
        if abs(logsum - logsum_old) < 0.001
            return logsum
        end
        logsum_old = logsum
        density *= 2
    end
end

## Contrib from Firefly.jl
using KernelDensity
function findpeaks(samples::AbstractVector)
    k = kde(samples)
    # the sign of the difference will tell use whether we are increasing or decreasing
    # using rle gives us the points at which the sign switches (local extreema)
    runs = rle(sign.(diff(k.density)))
    # if we start going up, first extreme will be maximum, else minimum
    start = runs[1][1] == 1 ? 1 : 2
    # find the peak indices at the local minima
    peak_idx = cumsum(runs[2])[start:2:end]
    sorted_idx = sortperm(k.density[peak_idx], rev = true)
    return k.x[peak_idx[sorted_idx]]
end



@testset "Sampling" begin include("sampling.jl") end
