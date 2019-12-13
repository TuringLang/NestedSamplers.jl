using NestedSamplers: randball

@testset "Rand Sphere" begin
    for _ in 1:100, k in 1:10
        x = randball(k)
        @test sum(x.^2) ≤ 1
    end
end
