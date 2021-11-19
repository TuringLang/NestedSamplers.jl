@doc raw"""
    Models.Eggbox()

Eggbox/Egg carton likelihood function

```math
z(x, y) = \left[a + \sin\frac{x}{b} + \sin\frac{x}{b} \right]^5
```

# Examples
```jldoctest
julia> model, lnZ = Models.Eggbox();

julia> lnZ
235.88
```
"""
function Eggbox()
    tmax = 5π

    # uniform prior from 0, 1
    prior(X) = X
    function loglike(X)
        a = cos(tmax * (2 * first(X) - 1) / 2)
        b = cos(tmax * (2 * last(X) - 1) / 2)
        return (2 + a * b)^5
    end

    lnZ = 235.88 # where do we get this from??
    return NestedModel(loglike, prior), lnZ
end
