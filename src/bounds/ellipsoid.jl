"""
    Bounds.Ellipsoid([T=Float64], N)
    Bounds.Ellipsoid(center::AbstractVector, A::AbstractMatrix)

An `N`-dimensional ellipsoid defined by

\$ (x - center)^T A (x - center) = 1 \$

where `size(center) == (N,)` and `size(A) == (N,N)`.
"""
mutable struct Ellipsoid{T} <: AbstractBoundingSpace{T}
    center::Vector{T}
    A::Matrix{T}
    axes::Matrix{T}
    axlens::Vector{T}
    volume::T
end


function Ellipsoid(center::AbstractVector, A::AbstractMatrix)
    axes, axlens = decompose(A)
    Ellipsoid(center, A, axes, axlens, _volume(A))
end
Ellipsoid(ndim::Integer) = Ellipsoid(Float64, ndim)
Ellipsoid(T::Type, ndim::Integer) = Ellipsoid(zeros(T, ndim), diagm(0 => ones(T, ndim)))
Ellipsoid{T}(center::AbstractVector, A::AbstractMatrix) where {T} = Ellipsoid(T.(center), T.(A))

Base.broadcastable(e::Ellipsoid) = (e,)

Base.ndims(ell::Ellipsoid) = length(ell.center)

# Returns the volume of an ellipsoid given its axes matrix
_volume(A::AbstractMatrix{T}) where {T} = T(volume_prefactor(size(A, 1))) / sqrt(det(A))
volume(ell::Ellipsoid) = ell.volume

# Returns the principal axes
axes(ell::Ellipsoid) = ell.axes

function decompose(A::AbstractMatrix)
    E = eigen(A)
    axlens = @. 1 / sqrt(E.values)
    axes = E.vectors .* axlens
    return axes, axlens
end

# axes and axlens
decompose(ell::Ellipsoid) = ell.axes, ell.axlens

# Scale to new volume
function scale!(ell::Ellipsoid, factor)
    # linear factor
    f = factor^(1 / ndims(ell))
    ell.A ./= f^2
    ell.axes .*= f
    ell.axlens .*= f
    ell.volume *= factor
    return ell
end

function endpoints(ell::Ellipsoid)
    axes, axlens = decompose(ell)
        # find major axis
    major_axis = axes[:, argmax(axlens)]
    return ell.center .- major_axis, ell.center .+ major_axis
end

function Base.in(x::AbstractVector, ell::Ellipsoid)
    d = x .- ell.center
    return d' * (ell.A * d) ≤ 1.0
end

randoffset(rng::AbstractRNG, ell::Ellipsoid{T}) where {T} = axes(ell) * randball(rng, T, ndims(ell))
Base.rand(rng::AbstractRNG, ell::Ellipsoid) = ell.center .+ randoffset(rng, ell)

fit(E::Type{<:Ellipsoid}, x::AbstractMatrix{S}; pointvol = 0) where {S} = fit(E{float(S)}, x; pointvol = pointvol)

function fit(E::Type{<:Ellipsoid{R}}, x::AbstractMatrix{S}; pointvol = 0) where {R,S}
    T = float(promote_type(R, S))
    x = T.(x)
    ndim, npoints = size(x)

    center, cov = mean_and_cov(x, 2)
    delta = x .- center

    # single element covariance will return NaN, but we want 0
    if npoints == 1
        cov = zeros(T, ndim, ndim)
    end

    # Covariance is smaller than r^2 by a factor of 1/(n+2)
    cov .*= ndim + 2

    # Ensure cov is nonsingular
    targetprod = (npoints * pointvol / volume_prefactor(ndim))^2
    make_eigvals_positive!(cov, targetprod)

    # edge case- single 0 is non-singular
    if size(cov) == (1, 1) && iszero(cov[1, 1])
        A = cov
    else
        A = inv(cov)
    end

    # calculate expansion factor necessary to bound each points
    f = diag(delta' * (A * delta))
    fmax = maximum(f)

    # try to avoid round-off errors s.t. furthest point obeys
    # x^T A x < 1 - √eps
    flex = 1 - sqrt(eps(T))
    if fmax > flex
        A .*= flex / fmax
    end
    ell = E(center[:, 1], A)

    if pointvol > 0
        minvol = npoints * pointvol
        vol = volume(ell)
        vol < minvol && scale!(ell, minvol / vol)
    end

    return ell
end


# ---------------------------------------------
# Helper functions

"""
    volume_prefactor(::Integer)

Volume constant for an n-dimensional sphere:

for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
"""
function volume_prefactor(n::Integer)
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

# sample N samples from unit D-dimensional ball
randball(T::Type, D::Integer, N::Integer) = randball(GLOBAL_RNG, T, D, N)
function randball(rng::AbstractRNG, T::Type, D::Integer, N::Integer)
    z = randn(rng, T, D, N)
    z .*= rand(rng, T, 1, N).^(1 ./ D) ./ sqrt.(sum(p->p^2, z, dims = 1))
    return z
end

# sample from unit D-dimensional ball
randball(T::Type, D::Integer) = randball(GLOBAL_RNG, T, D)
function randball(rng::AbstractRNG, T::Type, D::Integer)
    z = randn(rng, T, D)
    z .*= rand(rng)^(1 / D) / sqrt(sum(p->p^2, z))
    return z
end

function make_eigvals_positive!(cov::AbstractMatrix{T}, targetprod) where T
    ndims = size(cov, 1)
    for ntries in 1:100
        try
            cholesky(cov)
            E = eigen(cov)
            all(p -> p > 0 && isfinite(p), E.values) && return cov
        catch
            coeff = 1.1^(ntries) / 1.1^100
            cov .*= (1 - coeff)
            cov[diagind(cov)] .+= coeff
        end
    end
    @warn "Could not make ellipsoid axes singular; defaulting to unit-ball"
    cov .= diagm(0 => ones(T, ndims))
    return cov
end

make_eigvals_positive(cov::AbstractMatrix, targetprod) = make_eigvals_positive!(deepcopy(cov), targetprod)
