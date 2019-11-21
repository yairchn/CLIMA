using CLIMA.PlanetParameters
export LargeScaleSubsidence

abstract type Subsidence end

vars_state(::Subsidence, FT) = @vars()

"""
  LargeScaleSubsidence <: Subsidence
  Large Scale Subsidence Forcing as a non-diffusive flux 
"""
struct LargeScaleSubsidence{FT} <: Subsidence 
  "Large Scale Divergence"
  ls_div::FT
end

function flux_dry_subsidence!(::MoistureModel,flux::Grad, state::Vars, aux::Vars, t::Real)
  n = aux.orientation.∇Φ ./ norm(aux.orientation.∇Φ)
  D = sub.ls_div
  z = aux.orientation.Φ / grav 
  u_sub = - D * z
  ρ = state.ρ
  u = state.ρu / state.ρ
  flux.ρ   += ρ * -D * z
  flux.ρu  += ρ * (-D*z) * (-D*z)'
  flux.ρe  += -D*z * state.ρe
end

function flux_moist_subsidence!(moist::EquilMoist, flux::Grad, state::Vars, aux::Vars, t::Real)
  n = aux.orientation.∇Φ ./ norm(aux.orientation.∇Φ)
  D = sub.ls_div
  z = aux.orientation.Φ / grav 
  u_sub = - D * z
  ρ = state.ρ
  u = state.ρu / state.ρ
  flux.moisture.ρq_tot += -D*z * state.moisture.ρq_tot * ρ
end
