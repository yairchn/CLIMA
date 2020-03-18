
#### Tracer component in atmosphere model
abstract type TracerModel end

export NoTracers, Tracers

vars_state(::TracerModel, FT) = @vars()
vars_gradient(::TracerModel, FT) = @vars()
vars_diffusive(::TracerModel, FT) = @vars()
vars_aux(::TracerModel, FT) = @vars()

function atmos_nodal_update_aux!(
    ::TracerModel,
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function flux_tracers!(
    ::TracerModel,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
function diffusive!(::TracerModel, diffusive, ∇transform, state, aux, t)
    nothing
end
function flux_diffusive!(
    ::TracerModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    nothing
end
function gradvariables!(
    ::TracerModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

"""
    NoTracers <: TracerModel

No tracers
"""
struct NoTracers <: TracerModel end

"""
  Tracer <: TracerModel

Mechanism to include single tracer in an AtmosModel simulation
"""

struct Tracer <: TracerModel end

vars_state(tr::Tracer, FT) = @vars(ρχ::FT)
vars_gradient(::Tracer, FT) = @vars(χ::FT)
vars_diffusive(::Tracer, FT) = @vars(∇χ::SVector{3, FT})
vars_aux(::Tracer, FT) = @vars()

function gradvariables!(
    tracers::Tracer,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.tracers.χ = state.tracers.ρχ * ρinv
end

function diffusive!(
    tracers::Tracer,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    # diffusive flux
    diffusive.tracers.∇χ = ∇transform.tracers.χ
end

function flux_tracers!(
    tracers::Tracer,
    atmos::AtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    u = state.ρu / state.ρ
    flux.tracers.ρχ += state.tracers.ρχ * u
end

function flux_diffusive!(
    tracer::Tracer,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    D_t,
)
    d_χ = (-D_t) .* diffusive.tracer.∇χ
    flux_diffusive!(tracers, flux, state, d_χ)
end

function flux_diffusive!(tracers::Tracer, flux::Grad, state::Vars, d_χ)
    flux.tracers.ρχ += d_χ * state.ρ
end
