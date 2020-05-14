using NestedSamplers.Bounds: randball, volume_prefactor

@testset "Rand Sphere" begin
    for _ in 1:100, k in 1:10
        x = randball(Float64, k)
        @test sum(t->t^2, x) < 1
    end
end

@testset "Volume Prefactor" begin
    @test volume_prefactor(1) ≈ 2
    @test volume_prefactor(2) ≈ π
    @test volume_prefactor(3) ≈ 4 / 3 * π
    @test volume_prefactor(4) ≈ 1 / 2 * π^2
    @test volume_prefactor(5) ≈ 8 / 15 * π^2
    @test volume_prefactor(6) ≈ 1 / 6 * π^3
    @test volume_prefactor(7) ≈ 16 / 105 * π^3
    @test volume_prefactor(9) ≈ 32 / 945 * π^4
end
