#### Hyperdiffusion Model Functions
using DocStringExtensions
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export HyperDiffusion, NoHyperDiffusion, HorizontalHyperDiffusion

abstract type HyperDiffusion end
# Defaults
vars_state(::HyperDiffusion, FT) = @vars()
vars_aux(::HyperDiffusion, FT) = @vars()
vars_gradient(::HyperDiffusion, FT) = @vars()
vars_diffusive(::HyperDiffusion, FT) = @vars()
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars()
function atmos_init_aux!(::HyperDiffusion, ::AtmosModel, aux::Vars, geom::LocalGeometry) end
function atmos_nodal_update_aux!(::HyperDiffusion, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function gradvariables!(::HyperDiffusion, ::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real) end
function hyperdiffusive!(h::HyperDiffusion, hyperdiffusive::Vars, gradvars::Grad,
                         state::Vars, aux::Vars, t::Real) end
function flux_diffusive!(h::HyperDiffusion, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real) 
  nothing
end

struct NoHyperDiffusion <: HyperDiffusion end

"""
  HorizontalHyperDiffusion <: HyperDiffusion

"""
struct HorizontalHyperDiffusion <: HyperDiffusion end

vars_aux(::HorizontalHyperDiffusion, FT)                = @vars(Δ::FT)
vars_gradient(::HorizontalHyperDiffusion, FT)           = @vars(u_horz::SVector{3,FT}, h_tot::FT)
vars_gradient_laplacian(::HorizontalHyperDiffusion, FT) = @vars(u_horz::SVector{3,FT}, h_tot::FT)
vars_diffusive(::HorizontalHyperDiffusion, FT)          = @vars() 
vars_hyperdiffusive(::HorizontalHyperDiffusion, FT)     = @vars(∇³u_horz::SMatrix{3,3,FT,9}, ∇³h_tot::SVector{3,FT})
diffusive!(::HorizontalHyperDiffusion, _...)            = nothing

function atmos_init_aux!(::HyperDiffusion, ::AtmosModel, aux::Vars, geom::LocalGeometry) 
  aux.hyperdiffusion.Δ = lengthscale(geom)
end

function gradvariables!(::HorizontalHyperDiffusion, atmos::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  I = SDiagonal(1,1,1)
  k̂ = vertical_unit_vector(atmos.orientation,aux)
  u_horz = (I - k̂*k̂') * u
  transform.u_horz = u_horz
  transform.h_tot = total_specific_enthalpy(atmos.moisture, atmos.orientation, state, aux)
end

function hyperdiffusive!(h::HorizontalHyperDiffusion, hyperdiffusive::Vars, hypertransform::Grad,
                         state::Vars, aux::Vars, t::Real)
  ∇Δu_horz = hypertransform.u_horz
  ∇Δh_tot = hypertransform.h_tot
  #TODO update coefficient
  τ_timescale = 3600 * 8 
  ν₄ = (aux.hyperdiffusion.Δ/2)^4 / 2 / τ_timescale
  hyperdiffusive.∇³u_horz = ν₄ * ∇Δu_horz
  hyperdiffusive.∇³h_tot  = ν₄ * ∇Δh_tot
end

function flux_nondiffusive!(h::HorizontalHyperDiffusion, flux::Grad, state::Vars, aux::Vars, t::Real) end

function flux_diffusive!(h::HorizontalHyperDiffusion, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real) 
  flux.ρu += 1/state.ρ * hyperdiffusive.∇³u_horz
end
