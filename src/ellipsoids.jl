using StatsBase: mean_and_cov, sample
using Clustering
using Distributions: Categorical

# Helpers --------------------------------------------

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
        for i in 3:2:n
            f *= 2π / i
        end
    end
    return f
end

# sample from unit N-dimensional ball
function randball(rng::AbstractRNG, N::Integer)
    z = randn(rng, N)
    factor = rand(rng)^(1 / N) / sqrt(sum(z.^2))
    z .*= factor
    return z
end

randball(n::Integer) = randball(Random.GLOBAL_RNG, n)

function make_eigvals_positive!(cov::AbstractMatrix, targetprod)
    E = eigen(cov)
    mask = E.values .< 1e-10
    if any(mask)
        nzprod = prod(E.values[.!mask])
        nzeros = count(mask)
        E.values[mask] .= (targetprod / nzprod)^(1/nzeros)
        cov .= E.vectors * Diagonal(E.values) / E.vectors
    end
    return cov
end

make_eigvals_positive(cov::AbstractMatrix, targetprod) = make_eigvals_positive!(deepcopy(cov), targetprod)

# Ellipsoids --------------------------------------

abstract type AbstractEllipsoid end

"""
    Ellipsoid(center::AbstractVector, A::AbstractMatrix)

An `N`-dimensional ellipsoid defined by

\$ (x - center)^T A (x - center) = 1 \$

where `size(center) == (N,)` and `size(A) == (N,N)`.
"""
mutable struct Ellipsoid{T <: Number} <: AbstractEllipsoid
    center::Vector{T}
    A::Matrix{T}
    volume::T
end
# Unit ellipsoid
Ellipsoid(ndim::Integer) = Ellipsoid(zeros(ndim), diagm(0=>ones(ndim)))
Ellipsoid(center::AbstractVector, A::AbstractMatrix) =  Ellipsoid(center, A, _volume(A))

Base.broadcastable(e::Ellipsoid) = Ref(e)

Base.ndims(e::Ellipsoid) = length(e.center)

# Returns the volume of an ellipsoid given its axes matrix
_volume(A::AbstractMatrix) = unit_volume(size(A, 1)) / sqrt(det(A))
volume(ell::Ellipsoid) = ell.volume

# Returns the principal axes and their lengths
function decompose(ell::Ellipsoid)
    E = eigen(ell.A)
    axlens = @. 1 / sqrt(E.values)
    axes = E.vectors * Diagonal(axlens)
    return axes, axlens
end

# Scale to new volume
function scale!(ell::Ellipsoid, factor)
    f = factor^(1/ndims(ell))
    ell.A ./= f^2
    ell.volume = _volume(ell.A)
    return ell
end

function endpoints(ell::Ellipsoid)
    # get axes lengths
    E = eigen(ell.A)
    axlens = 1 ./ sqrt.(E.values)

    # get axes
    axes = E.vectors * Diagonal(axlens)
    
    # find major axis
    i = argmax(axlens)

    major_axis = axes[:, i]
    return ell.center .- major_axis, ell.center .+ major_axis
end

function Base.in(x::AbstractVector, ell::Ellipsoid)
    d = x .- ell.center
    return d' * ell.A * d ≤ 1.0
end

function Base.rand(rng::AbstractRNG, ell::Ellipsoid)
    # Generate random offset from center
    offset = decompose(ell)[1] * randball(rng, ndims(ell))

    return ell.center .+ offset
end

Base.rand(ell::Ellipsoid) = rand(Random.GLOBAL_RNG, ell)
Base.rand(rng::AbstractRNG, ell::Ellipsoid, n::Integer) = reduce(hcat, [rand(rng, ell) for _ in 1:n])
Base.rand(ell::Ellipsoid, n::Integer) = rand(Random.GLOBAL_RNG, ell, n)

function fit(::Type{Ellipsoid}, x::AbstractMatrix; pointvol=0.0, minvol=false)
    ndim, npoints = size(x)

    center, cov = mean_and_cov(x, 2)
    delta = x .- center

    # single element covariance will return NaN, but we want 0
    if npoints == 1 
        cov = zeros(eltype(center), ndim, ndim)
    end

    # Covariance is smaller than r^2 by a factor of 1/(n+2)
    cov .*= ndim + 2

    # Ensure cov is nonsingular
    targetprod = (npoints * pointvol / unit_volume(ndim))^2
    make_eigvals_positive!(cov, targetprod)

    A = inv(cov)

    # calculate expansion factor necessary to bound each points
    fmax = -Inf
    @inbounds for k in 1:npoints
        f = 0.0
        for j=1:ndim, i=1:ndim
            f += delta[i, k] * A[i, j] * delta[j, k]
        end
        fmax = max(fmax, f)
    end

    # try to avoid round-off errors s.t. furthest point obeys
    # x^T A x < 1 - √eps
    flex = 1 - sqrt(eps(eltype(A)))
    if fmax > flex
        A .*= flex / fmax
    end

    ell = Ellipsoid(reshape(center, ndim), A)

    if minvol
        v = npoints * pointvol
        volume(ell) < v && scale!(ell, v / volume(ell))
    end

    return ell
end

#-------------------------------------------
# MultiEllipsoid

struct MultiEllipsoid{T} <: AbstractEllipsoid
    ellipsoids::Vector{Ellipsoid{T}}
end

Base.broadcastable(me::MultiEllipsoid) = Ref(me)
Base.length(me::MultiEllipsoid) = length(me.ellipsoids)
Base.ndims(me::MultiEllipsoid) = ndims(first(me.ellipsoids))

volume(me::MultiEllipsoid) = sum(volume.(me.ellipsoids))

function scale!(me::MultiEllipsoid, factor)
    scale!.(me.ellipsoids, factor)
    return me
end

function fit(::Type{MultiEllipsoid}, x::AbstractMatrix; pointvol=0.0)
    parent = fit(Ellipsoid, x, pointvol=pointvol, minvol=true)
    ells = fit(MultiEllipsoid, x, parent, pointvol=pointvol)
    return MultiEllipsoid(ells)
end

function fit(::Type{MultiEllipsoid}, x::AbstractMatrix, parent::Ellipsoid; pointvol=0.0)
    ndim, npoints = size(x)

    # Clustering will fail with fewer than k=2 points
    npoints ≤ 2 && return [parent]

    p1, p2 = endpoints(parent)
    starting_points = hcat(p1, p2)
    R = kmeans!(x, starting_points; maxiter=10)
    labels = assignments(R)
    x1, x2 = [x[:, l .== labels] for l in unique(labels)] 

    # if either cluster has fewer than ndim points, it is ill-defined
    if size(x1, 2) < 2ndim || size(x2, 2) < 2ndim
        return [parent]
    end

    # Getting bounding ellipsoid for each cluster
    ell1, ell2 = fit.(Ellipsoid, (x1, x2), pointvol=pointvol, minvol=true)

    # If total volume decreased by over half, recurse
    if volume(ell1) + volume(ell2) < 0.5volume(parent)
        return vcat(fit(MultiEllipsoid, x1, ell1, pointvol=pointvol), 
                    fit(MultiEllipsoid, x2, ell2, pointvol=pointvol))
    end

    # Otherwise see if total volume is much larger than expected 
    # and split into more than 2 clusters
    if volume(parent) > 2npoints * pointvol
        out = vcat(fit(MultiEllipsoid, x1, ell1, pointvol=pointvol), 
                    fit(MultiEllipsoid, x2, ell2, pointvol=pointvol))
        sum(volume.(out)) < 0.5volume(parent) && return out
    end

    # Otherwise, return single bounding ellipse
    return [parent]
end

Base.in(x::AbstractVector, me::MultiEllipsoid) = any(Ref(x) .∈ me.ellipsoids)

function Base.rand(rng::AbstractRNG, me::MultiEllipsoid)   
    length(me) == 1 && return rand(rng, me.ellipsoids[1])

    # Select random ellipsoid
    vols = volume.(me.ellipsoids)
    idx = rand(rng, Categorical(vols ./ sum(vols)))
    ell = me.ellipsoids[idx]

    # Select point
    x = rand(rng, ell)

    # How many ellipsoids is the sample in
    n = count(Ref(x) .∈ me.ellipsoids)

    # Only accept with probability 1/n
    return n == 1 || rand(rng) < 1/n ? x : rand(rng, me)
end

Base.rand(me::MultiEllipsoid) = rand(Random.GLOBAL_RNG, me)
Base.rand(rng::AbstractRNG, me::MultiEllipsoid, n::Integer) = reduce(hcat, [rand(rng, me) for _ in 1:n])
Base.rand(me::MultiEllipsoid, n::Integer) = rand(Random.GLOBAL_RNG, me, n)
