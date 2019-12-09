using LinearAlgebra
using StatsBase: mean_and_cov

struct Ellipsoid{T <: Number}
    center::Vector{T}
    A::Matrix{T}
    invA::Matrix{T}
    volume::T
end

Ellipsoid(center::AbstractVector, A::AbstractMatrix) =  Ellipsoid(center, A, inv(A), _volume(A))

Base.ndims(e::Ellipsoid) = length(e.center)

function _volume(A::AbstractMatrix)
    ndim = size(A, 1)
    return unit_volume(ndim) / sqrt(det(A))
end

"""
    unit_volume(::Integer)

Volume constant for an n-dimensional sphere:

for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
"""
function unit_volume(n::Integer)
    if iseven(n)
        f = 1.0
        for i in 2:2:n
            f *= 2π / i
        end
    else
        f = 2.0
        for i in 3:3:n
            f *= (2.0 / i * π)
        end
    end
    return f
end

function randnball(n::Integer)
    z = randn(n)
    r2 = sum(z.^2)
    factor = rand()^(1 / n) / sqrt(r2)
    z .*= factor
    return z
end

function Base.rand(e::Ellipsoid)
    E = eigen(e.A)
    for j in 1:ndims(e)
        tmp = sqrt(E.values[j])
        E.vectors[:, j] .*= tmp
    end

    return E.vectors * randnball(ndims(e)) + e.center
end

function fit(::Type{Ellipsoid}, x::AbstractMatrix, enlarge = 1.0)
    ndim, npoints = size(x)
    center, A = mean_and_cov(x, 2)
    delta = x .- center
    iA = inv(A)
    fmax = -Inf
    for k in 1:npoints
        f = 0.0
        for i in 1:ndim, j in 1:ndim
            f += iA[i, j] * delta[i, k] * delta[j, k]
        end
        fmax = max(fmax, f)
    end

    fmax *= enlarge
    A .*= fmax
    iA .*= 1 / fmax
    return Ellipsoid(reshape(center, ndim), A)
end
