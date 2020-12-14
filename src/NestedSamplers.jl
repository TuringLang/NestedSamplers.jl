module NestedSamplers

# load submodules
include("bounds/Bounds.jl")
using .Bounds
include("proposals/Proposals.jl")
using .Proposals

       
using LinearAlgebra
using Random
using Random: AbstractRNG, GLOBAL_RNG

import AbstractMCMC: @ifwithprogresslogger,
                     AbstractSampler,
                     AbstractModel,
                     step,
                     bundle_samples,
                     mcmcsample,
                     save!!
using Distributions: quantile, UnivariateDistribution
using MCMCChains: Chains
import StatsBase
using StatsFuns: logaddexp,
                 log1mexp


export Bounds,
       Proposals,
       NestedModel,
       Nested,
       dlogz_convergence,
       decline_convergence

include("model.jl")         # The default model for nested sampling
include("staticsampler.jl") # The static nested sampler
include("step.jl")          # The stepping mechanics (extends AbstractMCMC)
include("sample.jl")        # Custom sampling (extends AbstractMCMC)

end
