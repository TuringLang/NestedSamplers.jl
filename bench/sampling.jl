# Full model sampling, follows prescription from jax-ns paper
using BenchmarkTools
using CSV
using NestedSamplers
using PythonCall
using Statistics
using StatsBase


rows = []
dims = [2, 4, 8, 16, 32]
for D in dims
    
    model, true_lnZ = Models.CorrelatedGaussian(D)
    splr = Nested(D, 50D; proposal=Proposals.Slice(), bounds=Bounds.Ellipsoid)
    # run once to extract values from state, also precompile
    ch, state = sample(model, splr; dlogz=0.01, chain_type=Array)
    lnZ = state.logz
    lnZstd = state.logzerr

    tt = @belapsed sample($model, $splr; dlogz=0.01, chain_type=Array)

    dlnZ = abs(true_lnZ - lnZ)

    row = (; library="NestedSamplers.jl", D, t=median(tt), lnZ, lnZstd, dlnZ)
    @info "$row"
    push!(rows, row)
end

path = joinpath(@__DIR__, "sampling_results.csv")
CSV.write(path, rows)
@info "output saved to $path"
