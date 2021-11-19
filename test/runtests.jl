using AbstractMCMC
using Distributions
using LinearAlgebra
using MCMCChains
using NestedSamplers
using StableRNGs
using StatsBase
using StatsFuns
using Test

rng = StableRNG(8425)
AbstractMCMC.setprogress!(get(ENV, "CI", "false") == "false")

include("utils.jl")

@testset "Deprecations" begin include("deprecations.jl") end
@testset "Bounds" begin include("bounds/bounds.jl") end
@testset "Proposals" begin include("proposals/proposals.jl") end
@testset "Sampler" begin include("sampler.jl") end
@testset "Sampling" begin include("sampling.jl") end
@testset "Models" begin include("models.jl") end
