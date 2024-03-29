using AbstractMCMC
using Distributions
using LinearAlgebra
using LogExpFunctions
using MCMCChains
using NestedSamplers
using StableRNGs
using StatsBase
using Test

rng = StableRNG(1234)
AbstractMCMC.setprogress!(get(ENV, "CI", "false") == "false")

include("utils.jl")

@testset "Deprecations" begin include("deprecations.jl") end
@testset "Bounds" begin include("bounds/bounds.jl") end
@testset "Proposals" begin include("proposals/proposals.jl") end
@testset "Sampler" begin include("sampler.jl") end
@testset "Sampling" begin include("sampling.jl") end
@testset "Models" begin include("models.jl") end
