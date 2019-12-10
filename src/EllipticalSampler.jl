using AbstractMCMC: AbstractSampler,
    AbstractSamplerState,
    VarInfo,
    Sampler,
    step!,
    Model,
    Transition,
    sample_init!,
    set_resume!,
    initialize_parameters

struct Nested{space} <: AbstractSampler 
    method::Symbol
    nactive::Integer
    maxiter::Integer
    enlarge::Float64
end

function Nested(nactive=100, maxiter=1000, enlarge=1.5; method=:single)
    if method ∉ [:single, :multi]
        error("Invalid Nested method $method")
    end
    return Nested{()}(method, nactive, maxiter, enlarge)
end

mutable struct NestedState{V<:VarInfo} <: AbstractSamplerState
    vi::V
    live_points_u::Matrix
    live_points_p::Matrix
    active_logl::Vector
    log_wt::Float64
end

NestedState(m::Model) = NestedState(VarInfo(m), zeros(0, 0), zeros(0, 0), zeros(0), 0.0)

function Sampler(alg::Nested, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = NestedState(model)
    return Sampler(alg, info, s, state)
end

function sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler,
    N::Integer;
    kwargs...
)
    set_resume!(spl; kwargs...)

    ndim = length(Turing.get_pvars(model))
    spl.state.active_points_u = rand(rng, ndim, spl.nactive)
    for i in 1:spl.nactive
        spl.state.active_logl = 
        spl.state.active_points_v = 
    end

    enlarge_linear = spl.enlarge^(1/ndim)

    pounts_u = rand()

end

function step!(::AbstractRNG, model::Model, spl::Sampler{<:Nested}, ::Integer; kwargs...)

    empty!(spl.state.vi)
    model(spl.state.vi, spl)
end

function Turing.step!(::AbstractRNG, model::Model, spl::Sampler{<:Nested}, ::Integer, ::Transition; kwargs...)

    empty!(spl.state.vi)
    model(spl.state.vi, spl)
end


transition_type(::Sampler{<:Nested}) = Transition

function assume(spl::Sampler{<:Nested}, dist::Distribution, vn::VarName, vi::VarInfo)
    x = rand(spl.nactive)
    v = 
end
