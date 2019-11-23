module Atmos

export AtmosModel

using LinearAlgebra, StaticArrays
using ..VariableTemplates
using ..MoistThermodynamics
using ..PlanetParameters
import ..MoistThermodynamics: internal_energy
using ..SubgridScaleParameters
using GPUifyLoops
using ..MPIStateArrays: MPIStateArray

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, vars_integrals, flux_nondiffusive!,
                        flux_diffusive!, source!, wavespeed, boundary_state!,
                        gradvariables!, diffusive!, init_aux!, init_state!,
                        update_aux!, integrate_aux!, LocalGeometry, lengthscale,
                        resolutionmetric, DGModel, num_integrals,
                        nodal_update_aux!, indefinite_stack_integral!,
                        reverse_indefinite_stack_integral!
using ..DGmethods.NumericalFluxes

"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling.

# Usage

    AtmosModel(orientation, ref_state, turbulence, moisture, radiation, source,
               boundarycondition, init_state)

"""
struct AtmosModel{O,RS,T,M,R,S,BC,IS} <: BalanceLaw
  orientation::O
  ref_state::RS
  turbulence::T
  moisture::M
  radiation::R
  source::S
  # TODO: Probably want to have different bc for state and diffusion...
  boundarycondition::BC
  init_state::IS
end

function vars_state(m::AtmosModel, FT)
  @vars begin
    ρ::FT
    ρu::SVector{3,FT}
    ρe::FT
    turbulence::vars_state(m.turbulence, FT)
    moisture::vars_state(m.moisture, FT)
    radiation::vars_state(m.radiation, FT)
  end
end
function vars_gradient(m::AtmosModel, FT)
  @vars begin
    u::SVector{3,FT}
    h_tot::FT
    turbulence::vars_gradient(m.turbulence,FT)
    moisture::vars_gradient(m.moisture,FT)
  end
end
function vars_diffusive(m::AtmosModel, FT)
  @vars begin
    ρτ::SHermitianCompact{3,FT,6}
    ρd_h_tot::SVector{3,FT}
    turbulence::vars_diffusive(m.turbulence,FT)
    moisture::vars_diffusive(m.moisture,FT)
  end
end


function vars_aux(m::AtmosModel, FT)
  @vars begin
    ∫dz::vars_integrals(m, FT)
    ∫dnz::vars_integrals(m, FT)
    coord::SVector{3,FT}
    orientation::vars_aux(m.orientation, FT)
    ref_state::vars_aux(m.ref_state,FT)
    turbulence::vars_aux(m.turbulence,FT)
    moisture::vars_aux(m.moisture,FT)
    radiation::vars_aux(m.radiation,FT)
  end
end
function vars_integrals(m::AtmosModel,FT)
  @vars begin
    radiation::vars_integrals(m.radiation,FT)
  end
end

"""
    flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars, aux::Vars,
                       t::Real)

Computes flux non-diffusive flux portion of `F` in:

```
∂Y
-- = - ∇ • (F_{adv} + F_{press} + F_{nondiff} + F_{diff}) + S(Y)
∂t
```
Where

 - `F_{adv}`      Advective flux             ; see [`flux_advective!`]@ref()
 - `F_{press}`    Pressure flux              ; see [`flux_pressure!`]@ref()
 - `F_{diff}`     Fluxes that state gradients; see [`flux_diffusive!`]@ref()
"""
@inline function flux_nondiffusive!(m::AtmosModel, flux::Grad, state::Vars,
                                    aux::Vars, t::Real)
  ρinv = 1/state.ρ
  ρu = state.ρu

  # primitive variable u⃗ velocity
  u = ρinv * ρu

  # advective terms (ρu⃗, ρu⃗⊗u⃗, ρu⃗e_tot)ᵀ
  flux.ρ   = ρu
  flux.ρu  = ρu .* u'
  flux.ρe  = u * state.ρe

  # moisture flux  (ρu⃗q_tot)
  flux_moisture!(m.moisture, flux, state, aux, t)

  # pressure terms (p𝐈, pu⃗)ᵀ
  p = pressure(m.moisture, m.orientation, state, aux)
  flux.ρu += p*I
  flux.ρe += u*p
  
  # radiation flux (ρFᵣ)
  flux_radiation!(m.radiation, flux, state, aux, t)
end

@inline function flux_diffusive!(m::AtmosModel, flux::Grad, state::Vars,
                                 diffusive::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu
  
  # (see `turbulence.jl` for shear stress tensor)
  ρτ = diffusive.ρτ
  ρd_h_tot = diffusive.ρd_h_tot
  flux.ρu += ρτ
  # diffusive momentum flux ρτ⃗
  flux.ρe += ρτ*u
  # diffusive enthalpy flux
  flux.ρe += ρd_h_tot
  # diffusive moisture fluxes (see `moisture.jl` for contributions)
  flux_diffusive!(m.moisture, flux, state, diffusive, aux, t)
end

@inline function wavespeed(m::AtmosModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu
  # maximum wavespeed = |n⃗⋅u⃗| + soundspeed
  return abs(dot(nM, u)) + soundspeed(m.moisture, m.orientation, state, aux)
end

function gradvariables!(atmos::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  # specify variables for which gradients are required 
  
  # primitive velocity vector u⃗ 
  transform.u = ρinv * state.ρu

  # total specific enthalpy  (h_tot = e_int + gz + 0.5(u⃗⋅u⃗) + RₘT)
  transform.h_tot = total_specific_enthalpy(atmos.moisture, atmos.orientation, state, aux)

  # subcomponent gradient terms 
  gradvariables!(atmos.moisture, transform, state, aux, t)
  gradvariables!(atmos.turbulence, transform, state, aux, t)
end


function symmetrize(X::StaticArray{Tuple{3,3}})
  SHermitianCompact(SVector(X[1,1], (X[2,1] + X[1,2])/2, (X[3,1] + X[1,3])/2, X[2,2], (X[3,2] + X[2,3])/2, X[3,3]))
end

function diffusive!(m::AtmosModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  ∇u = ∇transform.u
  # strain rate tensor
  S = symmetrize(∇u)
  # kinematic viscosity tensor
  ρν = dynamic_viscosity_tensor(m.turbulence, S, state, diffusive, ∇transform, aux, t)
  # momentum flux tensor
  diffusive.ρτ = scaled_momentum_flux_tensor(m.turbulence, ρν, S)

  ∇h_tot = ∇transform.h_tot
  # turbulent Prandtl number
  diag_ρν = ρν isa Real ? ρν : diag(ρν) # either a scalar or matrix
  # Diffusivity ρD_t = ρν/Prandtl_turb
  ρD_t = diag_ρν * inv_Pr_turb

  # diffusive flux of total enthalpy (ρ𝐝h_tot)
  diffusive.ρd_h_tot = -ρD_t .* ∇transform.h_tot

  # diffusive flux of total moisture (ρ𝐝q_tot)
  diffusive!(m.moisture, diffusive, ∇transform, state, aux, t, ρD_t)
  
  # diffusion terms required for SGS turbulence computations
  diffusive!(m.turbulence, diffusive, ∇transform, state, aux, t, ρD_t)
end

function update_aux!(dg::DGModel, m::AtmosModel, Q::MPIStateArray, t::Real)
  FT = eltype(Q)
  auxstate = dg.auxstate

  if num_integrals(m, FT) > 0
    indefinite_stack_integral!(dg, m, Q, auxstate, t)
    reverse_indefinite_stack_integral!(dg, m, auxstate, t)
  end

  nodal_update_aux!(atmos_nodal_update_aux!, dg, m, Q, t)
end

function atmos_nodal_update_aux!(m::AtmosModel, state::Vars, aux::Vars,
                                 diff::Vars, t::Real)
  # Update aux variables per timestep
  atmos_nodal_update_aux!(m.moisture, m, state, aux, t)
  atmos_nodal_update_aux!(m.radiation, m, state, aux, t)
  atmos_nodal_update_aux!(m.turbulence, m, state, aux, t)
end

function integrate_aux!(m::AtmosModel, integ::Vars, state::Vars, aux::Vars)
  # Integration step (vertical integration of specified aux variables. see aux.∫<vars>)
  integrate_aux!(m.radiation, integ, state, aux)
end

include("orientation.jl")
include("ref_state.jl")
include("turbulence.jl")
include("moisture.jl")
include("radiation.jl")
include("source.jl")
include("boundaryconditions.jl")
include("linear.jl")
include("remainder.jl")

# TODO: figure out a nice way to handle this
function init_aux!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  atmos_init_aux!(m.orientation, m, aux, geom)
  atmos_init_aux!(m.ref_state, m, aux, geom)
  atmos_init_aux!(m.turbulence, m, aux, geom)
end

"""
    source!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
Computes source terms `S(Y)` in:
```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(m::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  atmos_source!(m.source, m, source, state, aux, t)
end

boundary_state!(nf, m::AtmosModel, x...) =
  atmos_boundary_state!(nf, m.boundarycondition, m, x...)

# FIXME: This is probably not right....
boundary_state!(::CentralGradPenalty, bl::AtmosModel, _...) = nothing

function init_state!(m::AtmosModel, state::Vars, aux::Vars, coords, t)
  m.init_state(state, aux, coords, t)
end

end # module
