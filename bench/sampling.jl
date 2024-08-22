# Full model sampling, follows prescription from jax-ns paper
using BenchmarkTools
using CSV
using NestedSamplers
using ProgressLogging
using PythonCall
using StableRNGs
using Statistics
using StatsBase

dy = pyimport("dynesty")

rng = StableRNG(112358)

rows = []
dims = 2 .^ (1:4)
@progress name="NestedSamplers.jl" for D in dims
    model, true_lnZ = Models.CorrelatedGaussian(D)
    splr = Nested(D, 50D; proposal=Proposals.Slice(slices=5), bounds=Bounds.Ellipsoid)
    # run once to extract values from state, also precompile
    ch, state = sample(rng, model, splr; dlogz=0.01, chain_type=Array)
    t = @elapsed sample(rng, model, splr; dlogz=0.01, chain_type=Array)

    dlnZ = state.logz - true_lnZ

    row = (; library="NestedSamplers.jl", D, t=t, lnZ=state.logz, lnZstd=state.logzerr, dlnZ)
    @info "$row"
    push!(rows, row)
end

@progress name="dynesty" for D in dims
    model, true_lnZ = Models.CorrelatedGaussian(D)
    splr = dy.NestedSampler(
        model.prior_transform_and_loglikelihood.loglikelihood,
        model.prior_transform_and_loglikelihood.prior_transform,
        D; nlive=50D, bound="single", sample="slice", slices=5
    )
    t = @elapsed splr.run_nested(dlogz=0.01)
    res = splr.results
    lnZ = PyArray(res["logz"], copy=false)[end]
    lnZstd = PyArray(res["logzerr"], copy=false)[end]
    dlnZ = lnZ - true_lnZ

    row = (; library="dynesty", D, t, lnZ, lnZstd, dlnZ)
    @info "$row"
    push!(rows, row)
end

path = joinpath(@__DIR__, "sampling_results.csv")
CSV.write(path, rows)
@info "output saved to $path"
