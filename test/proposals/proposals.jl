const PROPOSALS = [
    Proposals.Uniform(),
    Proposals.RWalk(),
    Proposals.RStagger()
]

const BOUNDS = [
    Bounds.NoBounds(2),
    Bounds.Ellipsoid(2),
    Bounds.MultiEllipsoid(2)
]

@testset "interface - $(typeof(prop))" for prop in PROPOSALS, bound in BOUNDS
    logl(X) = -sum(x->x^2, X)
    prior(u) = 2u .- 1 # Uniform -1, to 1
    us = rand(2, 10)
    point, _bound = Bounds.rand_live(bound, us)
    loglstar = logl(prior(point))
    u, v, logL = prop(Random.GLOBAL_RNG, point, loglstar, _bound, logl, prior)
    # simple bounds checks
    @test all(x -> 0 < x < 1, u)
    @test all(x -> -1 < x < 1, v)
    
    # check new point actually has better likelihood
    @test logl(v) == logL ≥ loglstar
end

@testset "Uniform" begin
    # printing
    @test sprint(show, Proposals.Uniform()) == "NestedSamplers.Proposals.Uniform"
end

@testset "RWalk" begin
    prop = Proposals.RWalk()
    @test prop.scale == 1
    @test prop.ratio == 0.5
    @test prop.walks == 25

    @test_throws AssertionError Proposals.RWalk(ratio=-0.2)
    @test_throws AssertionError Proposals.RWalk(ratio=1.2)
    @test_throws AssertionError Proposals.RWalk(walks=0)
    @test_throws AssertionError Proposals.RWalk(walks=2, ratio=0.2)
    @test_throws AssertionError Proposals.RWalk(scale=-4)
end

@testset "RStagger" begin
    prop = Proposals.RStagger()
    @test prop.scale == 1
    @test prop.ratio == 0.5
    @test prop.walks == 25

    @test_throws AssertionError Proposals.RStagger(ratio=-0.2)
    @test_throws AssertionError Proposals.RStagger(ratio=1.2)
    @test_throws AssertionError Proposals.RStagger(walks=0)
    @test_throws AssertionError Proposals.RStagger(walks=2, ratio=0.2)
    @test_throws AssertionError Proposals.RStagger(scale=-4)
end

@testset "unitcheck" begin
    @test Proposals.unitcheck(rand(1000))
    @test !Proposals.unitcheck(randn(1000))

    # works with tuples, too
    @test Proposals.unitcheck((0.3, 0.6, 0.8))
end
