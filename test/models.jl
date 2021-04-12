const test_bounds = [Bounds.Ellipsoid, Bounds.MultiEllipsoid]
const test_props = [Proposals.Uniform(), Proposals.RWalk(ratio=0.9), Proposals.RStagger(ratio=0.9, walks=50), Proposals.Slice(slices=10), Proposals.RSlice()]


@testset "$(nameof(bound)), $(nameof(typeof(proposal)))" for bound in test_bounds, proposal in test_props
    @testset "Correlated Gaussian Conjugate Prior - ndims=$D" for D in [2, 5, 10]
        model, logz = Models.CorrelatedGaussian(D)

        sampler = Nested(D, 50 * (D + 1); bounds=bound, proposal=proposal)

        chain, state = sample(model, sampler; dlogz=0.01)
        chain_res = sample(chain, Weights(vec(chain[:weights])), length(chain))
        # test posteriors
        # vals = Array(chain_res)
        # means = mean(vals, dims=1)
        # tols = std(vals, mean=means, dims=1)
        # μ = fill(2.0, D)
        # Σ = fill(0.95, D, D)
        # Σ[diagind(Σ)] .= 1
        # expected = I * ((Σ + I) \ μ)
        # @test all(@.(abs(means - expected) < tols))

        # logz
        @test state.logz ≈ logz atol=2state.logzerr
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

        prior(X) = 10 .* X .- 5
        model = NestedModel(logl, prior)

        analytic_logz = log(4π * σ^2 / 100)


        spl = Nested(2, 1000, bounds=bound, proposal=proposal)
        chain, state = sample(rng, model, spl; dlogz=0.2)

        diff = state.logz - analytic_logz
        atol = 5state.logzerr
        if diff > atol
            @warn "logz estimate is poor" bound proposal error = diff tolerance = atol
        end

        @test state.logz ≈ analytic_logz atol = atol # within 3σ
        @test sort!(findpeaks(chain[:, 1, 1])[1:2]) ≈ [-1, 1] atol = 2σ
        @test sort!(findpeaks(chain[:, 2, 1])[1:2]) ≈ [-1, 1] atol = 2σ
    end
end

