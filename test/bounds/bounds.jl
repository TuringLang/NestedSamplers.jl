# Helper that returns a random N-dimensional ellipsoid
function random_ellipsoid(N::Integer)
    cov = rand(rng, N, N)
    while abs(det(cov)) < 1e-10
        cov = rand(rng, N, N)
    end
    return Ellipsoid(zeros(N), pinv(cov * cov'))
end

const BOUNDST = [
    Bounds.Ellipsoid,
    Bounds.MultiEllipsoid
]

@testset "interface - $B, $T, D=$D" for B in BOUNDST, T in [Float32, Float64], D in 1:20
    # creation, inspection
    bound = B(T, D)
    @test eltype(bound) == T
    @test ndims(bound) == D
    
    # sampling
    sample = rand(rng, bound)

    @test eltype(sample) == T
    @test size(sample) == (D,)
    @test sample ∈ bound
    
    nsamples = 1000
    samples = rand(rng, bound, nsamples)

    @test eltype(samples) == T
    @test size(samples) == (D, nsamples)
    @test all(samples[:, i] ∈ bound for i in axes(samples, 2))

    # fitting
    bound = Bounds.fit(B, samples)
    @test eltype(bound) == T
    @test all(samples[:, i] ∈ bound for i in axes(samples, 2))

    # robust fitting
    pv = Bounds.volume(bound) / size(samples, 2)
    bound2 = Bounds.fit(B, samples; pointvol = pv)
    @test Bounds.volume(bound2) ≈ Bounds.volume(bound) rtol = 1e-3
    
    # volume and scaling
    volfrac = 0.5
    bound_scaled = Bounds.scale!(deepcopy(bound), volfrac)
    @test Bounds.volume(bound) ≈ Bounds.volume(bound_scaled) / volfrac rtol = 1e-3

    # expected number of points that will fall within inner bound
    npoints = 5000
    expect = volfrac * npoints
    σ = sqrt((1 - volfrac) * expect)
    ninner = count(rand(rng, bound) ∈ bound_scaled for _ in 1:npoints)
    @test ninner ≈ expect atol = 3σ

    # printing
    @test sprint(show, bound) == "$(string(nameof(B))){$T}(ndims=$D)"

    # rand_live
    x = rand(rng, bound, 10)
    point, _bound = Bounds.rand_live(rng, bound, x)
    count(point ∈ x[:, i] for i in axes(x, 2)) == 1
    Btarget = B ∈ [Bounds.MultiEllipsoid] ? Bounds.Ellipsoid : B
    @test _bound isa Btarget
    @test point ∈ _bound && point ∈ bound
end

@testset "interface - NoBounds, $T, D=$D" for T in [Float16, Float32, Float64], D in 1:20
    # creation, inspection
    bound = Bounds.NoBounds(T, D)
    @test bound == Bounds.NoBounds{T}(D)
    @test eltype(bound) == T
    @test ndims(bound) == D
    
    # sampling
    sample = rand(rng, bound)

    @test eltype(sample) == T
    @test length(sample) == D
    @test sample ∈ bound
    
    samples = rand(rng, bound, 3)

    @test eltype(samples) == T
    @test size(samples) == (D, 3)
    @test all(samples[:, i] ∈ bound for i in axes(samples, 2))

    # fitting
    samples = randn(rng, T, D, 100)
    @test Bounds.fit(Bounds.NoBounds, samples) == bound
    @test eltype(bound) == T # matches eltype

    # robust fitting
    pv = 1 / size(samples, 2)
    bound_fit = Bounds.fit(Bounds.NoBounds, samples; pointvol = pv)
    @test bound_fit == bound
    @test Bounds.volume(bound_fit) == 1
    
    # volume and scaling
    bound_scaled = Bounds.scale!(deepcopy(bound), 0.5)
    @test bound_scaled == bound
    @test Bounds.volume(bound_scaled) == 1

    @test Bounds.axes(bound) == I
end

include("helpers.jl")
include("ellipsoids.jl")
