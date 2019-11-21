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
  u_sub = SVector{3,FT}(0,0,-D * z)
  flux.ρ   += ρ * u_sub
  flux.ρu  += ρ * (u_sub) * (u_sub)'
  flux.ρe  += u_sub * state.ρe
end

function flux_moist_subsidence!(moist::EquilMoist, flux::Grad, state::Vars, aux::Vars, t::Real)
  n = aux.orientation.∇Φ ./ norm(aux.orientation.∇Φ)
  D = sub.ls_div
  z = aux.orientation.Φ / grav 
  u_sub = - D * z
  ρ = state.ρ
  u = state.ρu / state.ρ
  flux.moisture.ρq_tot += u_sub * state.moisture.ρq_tot * ρ
end
