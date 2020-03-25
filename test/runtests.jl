using NestedSamplers
using Test
using Random
using LinearAlgebra
using IterTools

Random.seed!(8462852)

# Helper that returns a random N-dimensional ellipsoid
function random_ellipsoid(N::Integer)
    A = rand(N, N)
    while abs(det(A)) < 1e-10
        A = rand(N, N)
    end
    return Ellipsoid(zeros(N), A' * A)
end

# @testset "Ellipsoids" begin
#     include("helpers.jl")
#     include("ellipsoids.jl")
# end

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

include("sampling.jl")
