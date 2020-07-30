import StatsBase: ProbabilityWeights, pweights, Weights, weights

"""
    Particle(sample=missing, weight=missing, iteration=nothing)

A particle type containing pertinent information from the sampling routine. `weight` is the statistical weight of the sample. 

# Implements
The following functions for retrieving the samples from a vector of particles are implemented:
* `NestedSamplers.samples`

The following functions for retrieving the statistical weights of a vector of particles are
* `NestedSamplers.weights`
* `StatsBase.ProbabilityWeights`
* `StatsBase.pweights`
* `StatsBase.Weights` 
* `StatsBase.weights`
"""
struct Particle{T<:Union{Number,Missing},F<:Union{AbstractFloat,Missing}}
    sample::T
    weight::F
    iteration::Union{Int,Nothing}
end
Particle(sample, weight=missing, iteration=nothing) = Particle(sample, weight, iteration)
Particle() = Particle(missing, missing, nothing)

"""
    NestedSamplers.samples(particles)

Return the sample from each particle in the collection
"""
samples(particles) = map(p -> p.sample, particles)

"""
    NestedSamplers.weights(particles)

Return the sampling weight for each particle's sample in the collection
"""
weights(particles) = map(p -> p.weight, particles)

# StatsBase weights
for func in (:ProbabilityWeights, :pweights, :Weights, :weights)
    @eval begin
        $func(particles, args...) = $func(weights(collect(particles)), args...)
    end
end
