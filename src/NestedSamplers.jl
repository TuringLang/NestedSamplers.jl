module NestedSamplers

using LinearAlgebra
using Random

import AbstractMCMC: AbstractSampler,
                     AbstractModel,
                     step!,
                     sample_init!,
                     sample_end!,
                     bundle_samples,
                     mcmcsample
using Distributions
using MCMCChains: Chains
import StatsBase
using StatsFuns: logaddexp,
                 log1mexp

export NestedModel,
       Nested,
       dlogz_convergence,
       decline_convergence


include("bounds/Bounds.jl")        # The bounding algorithms
include("proposals/Proposals.jl")  # The proposal algorithms
include("model.jl")         # The default model for nested sampling
include("staticsampler.jl") # The static nested sampler
include("convergence.jl")   # The convergence callback methods
include("interface.jl")     # The interface to AbstractMCMC

end
