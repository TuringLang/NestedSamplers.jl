using NestedSamplers
using Test
using Random
using LinearAlgebra

Random.seed!(8462852)

# Helper that returns a random N-dimensional ellipsoid
function random_ellipsoid(N::Integer)
    A = rand(N, N)
    while  abs(det(A)) < 1e-10
        A = rand(N, N)
    end
    return Ellipsoid(zeros(N), A' * A)
end

@testset "Ellipsoids" begin
    include("helpers.jl")
    include("ellipsoids.jl")
end
