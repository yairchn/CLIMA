#### Hyperdiffusion Model Functions
using DocStringExtensions
using CLIMA.PlanetParameters
using CLIMA.SubgridScaleParameters
export HyperDiffusion, NoHyperDiffusion, HorizontalHyperDiffusion
export turbulence_tensors

abstract type HyperDiffusion end
# Defaults
vars_state(::HyperDiffusion, FT) = @vars()
vars_aux(::HyperDiffusion, FT) = @vars()
vars_gradient(::HyperDiffusion, FT) = @vars()
vars_diffusive(::HyperDiffusion, FT) = @vars()
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars()
function atmos_init_aux!(::HyperDiffusion, ::AtmosModel, aux::Vars, geom::LocalGeometry) end
function atmos_nodal_update_aux!(::HyperDiffusion, ::AtmosModel, state::Vars, aux::Vars, t::Real) end
function gradvariables!(::HyperDiffusion, transform::Vars, state::Vars, aux::Vars, t::Real) end
function hyperdiffusive!(h::HyperDiffusion, hyperdiffusive::Vars, gradvars::Grad,
                         state::Vars, aux::Vars, t::Real) end

struct NoHyperDiffusion <: HyperDiffusion end

struct HorizontalHyperDiffusion <: HyperDiffusion end

vars_aux(::HorizontalHyperDiffusion, FT) = @vars(D::SMatrix{3, 3, FT, 9})
vars_gradient(::HorizontalHyperDiffusion, FT) = @vars(ρu::SVector{3,FT})
vars_gradient_laplacian(::HorizontalHyperDiffusion, FT) = @vars(ρu::SVector{3,FT})
vars_diffusive(::HorizontalHyperDiffusion, FT) = @vars() 
vars_hyperdiffusive(::HorizontalHyperDiffusion, FT) = @vars(σ_horz::SMatrix{3,3,FT,9})
diffusive!(::HorizontalHyperDiffusion, _...) = nothing
function gradvariables!(::HorizontalHyperDiffusion, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.ρu = state.ρu
end
function hyperdiffusive!(h::HyperDiffusion, hyperdiffusive::Vars, hypertransform::Grad,
                         state::Vars, aux::Vars, t::Real)
  ∇Δρu = hypertransform.ρu
  #TODO update coefficient
  D = SDiagonal(1,1,1)
  hyperdiffusive.σ_horz = D * ∇Δρu
end
function flux_nondiffusive!(h::HyperDiffusion, flux::Grad, state::Vars, aux::Vars, t::Real) end
function flux_diffusive!(h::HyperDiffusion, flux::Grad, state::Vars,
                         diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real) 
  flux.ρu += 1/state.ρ * hyperdiffusive.σ_horz
end
