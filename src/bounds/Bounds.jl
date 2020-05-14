"""
    NestedSamplers.Bounds

This module contains the different algorithms for bounding the prior volume.

The available implementations are
* [`Bounds.NoBounds`](@ref) - no bounds on the prior volume (equivalent to a unit cube)
* [`Bounds.Ellipsoid`](@ref) - bound using a single ellipsoid
* [`Bounds.MultiEllipsoid`](@ref) - bound using multiple ellipsoids in an optimal cluster
"""
module Bounds

using LinearAlgebra
using Random: GLOBAL_RNG, AbstractRNG

using StatsBase: mean_and_cov
using Clustering
using Distributions: Categorical

export AbstractBoundingSpace


"""
    Bounds.AbstractBoundingSpace{T<:Number}

Abstract type for describing the bounding algorithms. For information about the interface, see the extended help (`??Bounds.AbstractBoundingSpace`)

# Extended Help

## Interface

The following functionality defines the interface for `AbstractBoundingSpace` for an example type `::MyBounds`

| Function | Required | Description |
|---------:|:--------:|:------------|
| `Base.rand(::RNG, ::MyBounds)` | x | Sample a single point from the prior volume |
| `Base.rand(::RNG, ::MyBounds, ::Int)` |  | Sample many points from the prior volume. Will simply repeat the singular version if not implemented. |
| `Base.in(point, ::MyBounds)` | x | Checks if the point is contained by the bounding space |
| `scale!(::MyBounds, factor)` | x | Scale the volume by the linear `factor`|
| `volume(::MyBounds)` | | Retrieve the current prior volume occupied by the bounds. |
| `fit(::Type{<:MyBounds}, points, pointvol=0)` | x | update the bounds given the new `points` each with minimum volume `pointvol`|
"""
abstract type AbstractBoundingSpace{T <: Number} end

Base.eltype(::AbstractBoundingSpace{T}) where {T} = T

# convenience
Base.rand(B::AbstractBoundingSpace) = rand(GLOBAL_RNG, B)
Base.rand(B::AbstractBoundingSpace, N::Integer) = rand(GLOBAL_RNG, B, N)

# fallback method
Base.rand(rng::AbstractRNG, B::AbstractBoundingSpace, N::Integer) = reduce(hcat, [rand(rng, B) for _ in 1:N])

function Base.show(io::IO, bound::B) where {T,B <: AbstractBoundingSpace{T}}
    base = nameof(B) |> string
    print(io, "$base{$T}(ndims=$(ndims(bound)))")

    return nothing
end

# ---------------------------------------------------

"""
    Bounds.NoBounds([T=Float64], N)

Unbounded prior volume; equivalent to the unit cube in `N` dimensions.
"""
struct NoBounds{T} <: AbstractBoundingSpace{T}
    ndims::Int
end
NoBounds(D::Integer) = NoBounds{Float64}(D)
NoBounds(T::Type, D::Integer) = NoBounds{T}(D)

Base.ndims(B::NoBounds) = B.ndims

Base.rand(rng::AbstractRNG, b::NoBounds{T}) where {T} = rand(rng, T, ndims(b))
Base.rand(rng::AbstractRNG, b::NoBounds{T}, N::Integer) where {T} = rand(rng, T, ndims(b), N)
Base.in(pt, ::NoBounds) = all(0 .< pt .< 1)
fit(::Type{<:NoBounds}, points::AbstractMatrix{T}; kwargs...) where {T} = NoBounds(T, size(points, 1))
scale!(b::NoBounds, factor) = b
volume(::NoBounds{T}) where {T} = one(T)


include("ellipsoid.jl")
include("multiellipsoid.jl")

end # module Bounds