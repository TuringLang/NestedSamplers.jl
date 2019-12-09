struct Elliptical{space} <: InferenceAlgorithm 
    maxiter::Integer
end

Elliptical(maxiter) = Elliptical{()}(maxiter)

mutable struct EllipticalState{V<:VarInfo} <: AbstractSamplerState
    vi::V
    log_wt
end

EllipticalState(m::Model) = EllipticalState(VarInfo(m), 0.0)
