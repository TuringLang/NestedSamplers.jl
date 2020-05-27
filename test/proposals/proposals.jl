const PROPOSALS = [
    Proposals.Uniform(),
    Proposals.RWalk()
]

const BOUNDS = [
    Bounds.NoBounds(2),
    Bounds.Ellipsoid(2),
    Bounds.MultiEllipsoid(2)
]

@testset "interface" for P in PROPOSALS, bound in BOUNDS
    logl(X) = -sum(x->x^2, X)
    prior(u) = 2u .- 1 # Uniform -1, to 1
    loglstar = -25
    us = rand(2, 10)
    point, _bound = Bounds.rand_live(bound, us)
    u, v, logL = P(Random.GLOBAL_RNG, point, loglstar, _bound, logl, prior)
    # simple bounds checks
    @test all(x -> 0 < x < 1, u)
    @test all(x -> -1 < x < 1, v)
    @test logL > loglstar
    
    # check new point actually has better likelihood
    @test logl(v) == logL ≥ -25
end

@testset "Uniform" begin
    # printing
    @test sprint(show, Proposals.Uniform()) == "NestedSamplers.Proposals.Uniform"
end

@testset "RWalk" begin
end
