using NestedSamplers: randball, unit_volume

@testset "Rand Sphere" begin
    for _ in 1:100, k in 1:10
        x = randball(k)
        @test sum(x.^2) < 1
    end
end

@testset "Volume Prefactor" begin
    @test unit_volume(1) ≈ 2
    @test unit_volume(2) ≈ π
    @test unit_volume(3) ≈ 4/3 * π
    @test unit_volume(4) ≈ 1/2 * π^2
    @test unit_volume(5) ≈ 8/15 * π^2
    @test unit_volume(6) ≈ 1/6 * π^3
    @test unit_volume(7) ≈ 16/105 * π^3
    @test unit_volume(9) ≈ 32/945 * π^4
end
