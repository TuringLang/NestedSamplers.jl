using NestedSamplers: Ellipsoid, MultiEllipsoid, fit, scale!

const NMAX = 20

@testset "Spheres: N=$N" for N in 1:NMAX  
    scale = 5
    center = 2scale .* ones(N)
    A = ones(N) ./ scale^2

end