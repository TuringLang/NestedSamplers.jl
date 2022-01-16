# Full model sampling, follows prescription from jax-ns paper
using BenchmarkTools
using CSV
using NestedSamplers
using ProgressLogging
using StableRNGs
using Statistics
using StatsBase

rng = StableRNG(112358)

rows = []
dims = 2 .^ (1:5)
@progress for D in dims
    model, true_lnZ = Models.CorrelatedGaussian(D)
    splr = Nested(D, 50D; proposal=Proposals.Slice(slices=5), bounds=Bounds.Ellipsoid)
    # run once to extract values from state, also precompile
    ch, state = sample(rng, model, splr; dlogz=0.01, chain_type=Array)
    t = @belapsed sample($rng, $model, $splr; dlogz=0.01, chain_type=Array)

    dlnZ = state.logz - true_lnZ

    row = (; library="NestedSamplers.jl", D, t=t, lnZ=state.logz, lnZstd=state.logzerr, dlnZ)
    @info "$row"
    push!(rows, row)
end

path = joinpath(@__DIR__, "sampling_results.csv")
CSV.write(path, rows)
@info "output saved to $path"
