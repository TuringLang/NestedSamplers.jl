module NestedSamplers

include("ellipsoids.jl")

function logaddexp(loga, logb)
    loga == logb && return loga + log(2)

    tmp = loga - logb
    return loga + log1p(exp(-abs(tmp)))
end

include("EllipticalSampler.jl")

end
