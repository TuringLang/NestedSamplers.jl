"""
This module contains various statistical models in the form of [`NestedModel`](@ref)s. These models can be used for examples and for testing.

* [`Models.GaussianShells`](@ref)
* [`Models.CorrelatedGaussian`](@ref)
"""
module Models

using ..NestedSamplers

using Distributions
using LinearAlgebra
using LogExpFunctions

include("shells.jl")
include("correlated.jl")

end # module
