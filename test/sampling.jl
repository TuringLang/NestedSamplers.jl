using Distributions
using AbstractMCMC
using MCMCChains: Chains
using StatsFuns
using StatsBase

@testset "Bundles" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chains, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains, 2) == size(val_arr, 2)

    # test with add_live = false
    chains2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, chain_type=Array)
    
    @test size(chains2, 2) == size(val_arr2, 2)
    @test size(chains2, 1) < size(chains, 1) && size(val_arr2, 1) < size(val_arr, 1)

    # test check_wsum kwarg
    chains3, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr3, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains3, 2) == size(val_arr3, 2)
end

@testset "Stopping criterion" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    
    chains, state = sample(rng, model, spl; add_live=false, dlogz=1.0)
    logz_remain = maximum(state.logl) + state.logvol
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    @test delta_logz < 1.0

    chains, state = sample(rng, model, spl; add_live=false, maxiter=3)
    @test state.it < 3

    chains, state = sample(rng, model, spl; add_live=false, maxcall=10)
    @test state.ncall < 10

    chains, state = sample(rng, model, spl; add_live=false, maxlogl=0.2)
    @test state.logl[1] > 0.2
end

const test_bounds = [Bounds.NoBounds, Bounds.Ellipsoid, Bounds.MultiEllipsoid]
const test_props = [Proposals.Uniform(), Proposals.RWalk(ratio=0.9), Proposals.RStagger(ratio=0.9, walks=75), Proposals.Slice(slices=10), Proposals.RSlice()]

# @testset "Flat - $(nameof(bound)), $(nameof(typeof(proposal)))" for bound in test_bounds, proposal in test_props
#     logl(::AbstractVector{T}) where T = zero(T)
#     priors = [Uniform(0, 1)]
#     model = NestedModel(logl, priors)

#     spl = Nested(1, 4, bound=bound, proposal=proposal)
#     chain, state = sample(model, spl, dlogz = 0.2)

#     @test state.logz ≈ 0 atol = 1e-9 # TODO
#     @test state.h ≈ 0 atol = 1e-9 # TODO
# end

@testset "Gaussian - $(nameof(bound)), $(nameof(typeof(proposal)))" for bound in test_bounds, proposal in test_props
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
    chain, state = sample(rng, model, spl)

    diff = state.logz - analytic_logz
    atol = bound <: Bounds.NoBounds ? 5state.logzerr : 3state.logzerr
    if diff > atol
        @warn "logz estimate is poor" bound proposal error = diff tolerance = atol
    end


    @test state.logz ≈ analytic_logz atol = atol # within 3σ
    @test sort!(findpeaks(chain[:, 1, 1])[1:2]) ≈ [-1, 1] atol = 2σ
    @test sort!(findpeaks(chain[:, 2, 1])[1:2]) ≈ [-1, 1] atol = 2σ
end
