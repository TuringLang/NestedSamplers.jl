module NestedSamplers

# load submodules
include("bounds/Bounds.jl")
using .Bounds
include("proposals/Proposals.jl")
using .Proposals

       
using LinearAlgebra
using Random
using Random: AbstractRNG, GLOBAL_RNG

import AbstractMCMC: AbstractSampler,
                        AbstractModel,
                        step!,
                        sample_init!,
                        sample_end!,
                        bundle_samples,
                        mcmcsample
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

include("particle.jl")
include("model.jl")         # The default model for nested sampling
include("staticsampler.jl") # The static nested sampler
include("convergence.jl")   # The convergence callback methods
include("interface.jl")     # The interface to AbstractMCMC

end
