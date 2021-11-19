"""
    Models.GaussianShells()

2-D Gaussian shells centered at `[-3.5, 0]` and `[3.5, 0]` with a radius of 2 and a shell width of 0.1

# Examples
```jldoctest
julia> model, lnZ = Models.GaussianShells();

julia> lnZ
-1.75
```
"""
function GaussianShells()
    μ1 = [-3.5, 0]
    μ2 = [3.5, 0]

    prior(X) = 12 .* X .- 6
    loglike(X) = logaddexp(logshell(X, μ1), logshell(X, μ2))

    lnZ = -1.75
    return NestedModel(loglike, prior), lnZ
end

function logshell(X, μ, radius=2, width=0.1)
    d = LinearAlgebra.norm(X - μ)
    norm = -log(sqrt(2 * π * width^2))
    return norm - (d - radius)^2 / (2 * width^2)
end
