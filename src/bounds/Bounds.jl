"""
    NestedSamplers.Bounds

This module contains the different algorithms for bounding the prior volume.

The current implementations are
* [`Bounds.NoBounds`](@ref) - no bounds on the prior volume (equivalent to a unit cube)
* [`Bounds.Ellipsoid`](@ref) - bound using a single ellipsoid
* [`Bounds.MultiEllipsoid`](@ref) - bound using multiple ellipsoids in an optimal cluster
"""
module Bounds

"""
    Bounds.AbstractBoundingSpace{T<:Number,D<:Integer}

Abstract type for describing the bounding algorithms with `D` dimensions. For information about the interface, see the extended help (`??Bounds.AbstractBoundingSpace`)

# Extended Help

## Interface

The following functionality defines the interface for `AbstractBoundingSpace` for an example type `::MyBounds`

| Function | Required | Description |
|---------:|:--------:|:------------|
| `Base.rand(::MyBounds)` | x | Sample a single point from the prior volume |
| `Base.rand(::MyBounds, ::Int)` |  | Sample many points from the prior volume. Will simply repeat the singular version if not implemented. |
| `randoffset(::MyBounds)` | x | Generate a point randomly outward from the center of the bounding space
| `Base.in(::MyBounds, point)` | x | Checks if the point is contained by the bounding space
| `scale!(::MyBounds, factor)` | x | Scale the volume by `factor`|
| `span(::MyBounds)` | | If applicable, retrieve the span of the prior volume. For example, the principal axes of an ellipsoid.
| `volume(::MyBounds)` | | If applicable, retrieve the current prior volume occupied by the bounds.
| `fit!(::MyBounds, points, pointvol=0)` | x | update the bounds given the new `points` each with minimum volume `pointvol`.
"""
abstract type AbstractBoundingSpace{T <: Number,D <: Integer} end

struct NoBounds{T,D} <: AbstractBoundingSpace{T,D} end

Base.rand(::NoBounds{T,D}) where {T,D} = rand(T, D)

end # module Bounds
