# Helper that returns a random N-dimensional ellipsoid
function random_ellipsoid(N::Integer)
    A = rand(N, N)
    while abs(det(A)) < 1e-10
        A = rand(N, N)
    end
    return Ellipsoid(zeros(N), A * A')
end

include("helpers.jl")
include("ellipsoids.jl")
