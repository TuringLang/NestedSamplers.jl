using NestedSamplers: Particle, weights, samples

@testset "Interface" begin
    p = Particle()
    @test p isa Particle{Missing,Missing}
    @test p.sample === p.weight === missing
    @test p.iteration === nothing
end

@testset "Interface - $T" for T in (Int32, Int64, Float32, Float64)
    p = Particle(rand(rng, T))
    @test p isa Particle{T,Missing}
    @test p.weight === missing
    @test p.iteration === nothing

    @testset "Interface - $T,$F" for F in (Float32, Float64)
        p = Particle(rand(rng, T), rand(rng, F))
        @test p isa Particle{T,F}
        @test p.iteration === nothing

        p = Particle(rand(rng, T), rand(rng, F), 8)
        @test p isa Particle{T,F}
        @test p.iteration == 8
    end
end

@testset "samples and weights"  begin
    samps = randn(rng, 100)
    ws = rand(rng, 100)
    parts = Particle.(samps, ws, 1:100)

    @test weights(parts) == ws
    @test samples(parts) == samps
end
