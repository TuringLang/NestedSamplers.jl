module NestedSamplers
using Turing

include("ellipsoids.jl")

function logaddexp(loga, logb)
    loga == logb && return loga + log(2)

    tmp = loga - logb
    s = sign(tmp)
    return loga + log1p(exp(-s * tmp))
end


end
