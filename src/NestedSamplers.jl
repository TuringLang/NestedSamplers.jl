module NestedSamplers

# load submodules
include("bounds/Bounds.jl")
include("proposals/Proposals.jl")

export Bounds,
       Proposals
    #     NestedModel,
    #    Nested,
    #    dlogz_convergence,
    #    decline_convergence,
       
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
using Distributions
using MCMCChains: Chains
import StatsBase
using StatsFuns: logaddexp,
                    log1mexp


# include("model.jl")         # The default model for nested sampling
# include("staticsampler.jl") # The static nested sampler
# include("convergence.jl")   # The convergence callback methods
# include("interface.jl")     # The interface to AbstractMCMC

end
