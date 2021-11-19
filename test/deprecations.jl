
@testset "Proposals.Uniform -> Proposals.Rejection deprecation" begin
    prop = @test_deprecated Proposals.Uniform()
    @test prop === Proposals.Rejection()
end