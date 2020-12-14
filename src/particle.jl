import StatsBase: ProbabilityWeights, pweights, Weights, weights

"""
    Particle(sample=missing, weight=missing, iteration=nothing)

A particle type containing pertinent information from the sampling routine. `weight` is the statistical weight of the sample.
"""
struct Particle{T<:Union{Number,Missing},F<:Union{AbstractFloat,Missing}}
    sample::T
    weight::F
    iteration::Union{Int,Nothing}
end
Particle(sample, weight=missing, iteration=nothing) = Particle(sample, weight, iteration)
Particle() = Particle(missing, missing, nothing)
