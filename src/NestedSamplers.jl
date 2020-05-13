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


include("bounds.jl")        # The bounding algorithms
include("proposals.jl")     # The proposal algorithms
include("ellipsoids.jl")    # The actual math
include("staticsampler.jl") # The sampler, model, and transition types
include("convergence.jl")   # The convergence methods
include("interface.jl")     # The interface to AbstractMCMC

end
