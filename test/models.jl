const test_bounds = [
    Bounds.Ellipsoid,
    Bounds.MultiEllipsoid
]
const test_props = [
    Proposals.Rejection(maxiter=Int(1e6)),
    Proposals.RWalk(ratio=0.5, walks=50),
    Proposals.RStagger(ratio=0.5, walks=50),
    Proposals.Slice(slices=10),
    Proposals.RSlice(slices=10)
]

const MAXZSCORES = Dict(zip(
    Iterators.product(test_bounds, test_props),
    [3, 3, 5, 6, 6, 3, 5, 7, 4, 3]
))

function test_logz(measured, actual, error, bound, proposal)
    diff = measured - actual
    zscore = abs(diff) / error
    @test measured ≈ actual atol=MAXZSCORES[(bound, proposal)] * error
end


@testset "$(nameof(bound)), $(nameof(typeof(proposal)))" for bound in test_bounds, proposal in test_props
    @testset "Correlated Gaussian Conjugate Prior - ndims=$D" for D in [2, 4]
        model, logz = Models.CorrelatedGaussian(D)
        # match JAXNS paper setup, generally
        sampler = Nested(D, 50D; bounds=bound, proposal=proposal)

        chain, state = sample(rng, model, sampler; dlogz=0.01)
        chain_res = sample(rng, chain, Weights(vec(chain[:weights])), length(chain))
        # test posteriors
        vals = Array(chain_res)
        means = mean(vals, dims=1)
        tols = 2std(vals, mean=means, dims=1) # 2-sigma
        μ = fill(2.0, D)
        Σ = fill(0.95, D, D)
        Σ[diagind(Σ)] .= 1
        expected = Σ * ((Σ + I) \ μ)
        @test all(@.(abs(means - expected) < tols))

        # logz
        test_logz(state.logz, logz, state.logzerr, bound, proposal)
    end

    @testset "Gaussian Shells" begin
        model, logz = Models.GaussianShells()

        sampler = Nested(2, 1000; bounds=bound, proposal=proposal)

        chain, state = sample(rng, model, sampler; dlogz=0.01)

        # logz
        test_logz(state.logz, logz, state.logzerr, bound, proposal)
    end

    @testset "Gaussian Mixture Model" begin
        σ = 0.1
        μ1 = ones(2)
        μ2 = -ones(2)
        inv_σ = diagm(0 => fill(1 / σ^2, 2))

        function logl(x)
            dx1 = x .- μ1
            dx2 = x .- μ2
            f1 = -dx1' * (inv_σ * dx1) / 2
            f2 = -dx2' * (inv_σ * dx2) / 2
            return logaddexp(f1, f2)
        end

        prior(X) = muladd.(10, X, -5)
        model = NestedModel(logl, prior)

        analytic_logz = log(4π * σ^2 / 100)


        spl = Nested(2, 1000, bounds=bound, proposal=proposal)
        chain, state = sample(rng, model, spl; dlogz=0.01)
        chain_res = sample(rng, chain, Weights(vec(chain[:weights])), length(chain))

        test_logz(state.logz, analytic_logz, state.logzerr, bound, proposal)

        xmodes = sort!(findpeaks(chain_res[:, 1, 1])[1:2])
        @test xmodes[1] ≈ -1 atol = σ
        @test xmodes[2] ≈ 1 atol = σ
        ymodes = sort!(findpeaks(chain_res[:, 2, 1])[1:2])
        @test ymodes[1] ≈ -1 atol = σ
        @test ymodes[2] ≈ 1 atol = σ
    end

    @testset "Eggbox" begin
        model, logz = Models.Eggbox()

        sampler = Nested(2, 1000; bounds=bound, proposal=proposal)

        chain, state = sample(rng, model, sampler; dlogz=0.1)

        test_logz(state.logz, logz, state.logzerr, bound, proposal)

        chain_res = sample(rng, chain, Weights(vec(chain[:weights])), length(chain))
        xmodes = sort!(findpeaks(chain_res[:, 1, 1])[1:5])
        @test all(isapprox.(xmodes, 0.1:0.2:0.9, atol=0.2))
        ymodes = sort!(findpeaks(chain_res[:, 2, 1])[1:5])
        @test all(isapprox.(ymodes, 0.1:0.2:0.9, atol=0.2))
    end
end
