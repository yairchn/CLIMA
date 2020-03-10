using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: BalanceLaw,
                        vars_aux,
                        vars_state,
                        vars_gradient,
                        vars_diffusive,
                        flux_nondiffusive!,
                        flux_diffusive!,
                        source!,
                        wavespeed,
                        boundary_state!,
                        update_aux!,
                        update_aux_diffusive!,
                        gradvariables!,
                        init_aux!,
                        init_state!,
                        LocalGeometry,
                        DGModel,
                        nodal_update_aux!,
                        diffusive!,
                        create_state,
                        calculate_dt,
                        reverse_integral_set_aux!

# ------------------------ Description ------------------------- #
# The test is based on a modelling set-up designed for the
# 8th International Cloud Modelling Workshop
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See chapter 2 in Arabas et al 2015 for setup details:
#@Article{gmd-8-1677-2015,
#AUTHOR = {Arabas, S. and Jaruga, A. and Pawlowska, H. and Grabowski, W. W.},
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk, and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}
# ------------------------ Description ------------------------- #

struct KinematicModel{FT} <: BalanceLaw
  wmax::FT
  xmax::FT
  zmax::FT
end

vars_state(m::KinematicModel, FT) = @vars()
vars_gradient(m::KinematicModel, FT) = @vars()
vars_diffusive(m::KinematicModel, FT) = @vars()
vars_aux(m::KinematicModel, FT) = @vars()
vars_integrals(m::KinematicModel, FT) = @vars()
vars_reverse_integrals(m::KinematicModel, FT) = @vars()

function init_aux!(m::KinematicModel, aux::Vars, geom::LocalGeometry)
end

@inline function flux_nondiffusive!(m::KinematicModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end

function gradvariables!(m::KinematicModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

function diffusive!(m::KinematicModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
end

@inline function flux_diffusive!(m::KinematicModel, flux::Grad, state::Vars, diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
end

@inline function flux_diffusive!(m::KinematicModel, flux::Grad, state::Vars, τ, d_h_tot)
end

@inline function wavespeed(m::KinematicModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu
  return abs(dot(nM, u))
end

# ------------------------------------------------------------------ BOILER PLATE :)
# function update_aux!(dg::DGModel, m::KinematicModel, Q::MPIStateArray, t::Real)
#   nodal_update_aux!(atmos_nodal_update_aux!, dg, m, Q, t)
#   return true
# end
function integral_load_aux!(m::KinematicModel, integ::Vars, state::Vars, aux::Vars) end
function integral_set_aux!(m::KinematicModel, aux::Vars, integ::Vars) end
function reverse_integral_load_aux!(m::KinematicModel, integ::Vars, state::Vars, aux::Vars) end
function reverse_integral_set_aux!(m::KinematicModel, aux::Vars, integ::Vars) end
# ------------------------------------------------------------------


function init_saturation_adjustment!(bl, state, aux, (x,y,z), t)
  FT = eltype(Q)

  # initial condition
  θ_0::FT    = 289         # K
  p_0::FT    = 101500      # Pa
  p_1000::FT = 100000      # Pa
  qt_0::FT   = 7.5 * 1e-3  # kg/kg
  z_0::FT    = 0           # m
  q_pt_0 = PhasePartition(qt_0)

  R_m, cp_m, cv_m, γ = gas_constants(q_pt_0)

  # Pressure profile assuming hydrostatic and constant θ and qt profiles.
  # It is done this way to be consistent with Arabas paper.
  # It's not necessarily the best way to initialize with our model variables.
  p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
              R_d / cp_d * grav / θ_0 / R_m * (z - z_0)
             )^(cp_d / R_d)
  T::FT = θ_0 * exner_given_pressure(p, q_pt_0)
  ρ::FT = p / R_m / T

  # TODO should this be more "grid aware"?
  # the velocity is calculated as derivative of streamfunction
  ρu::FT = bl.wmax * bl.xmax/bl.zmax * cos(π * z/bl.zmax) * cos(2*π * x/bl.xmax)
  ρw::FT = 2*bl.wmax * sin(π * z/bl.zmax) * sin(2*π * x/bl.xmax)
  u = ρu / ρ
  w = ρw / ρ

  ρq_tot::FT = ρ * qt_0

  e_int  = internal_energy(T, q_pt_0)
  ρe_tot = ρ * (grav * z + (1//2)*(u^2 + w^2) + e_int)

  state.ρ = ρ
  state.ρu = SVector(ρu, FT(0), ρw)
  state.ρe = ρe_tot
  state.ρq_tot = ρq_tot
  return nothing
end

function config_saturation_adjustment(FT, N, resolution, xmax, ymax, zmax)

  # Boundary conditions
  bc = NoFluxBC()

  # Choose explicit solver
  ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  wmax = FT(.6)
  # Set up the model
  model = KinematicModel{FT}(FT(wmax), # m/s
                             FT(xmax), # m
                             FT(zmax)  # m
                             )

        # (Δx, Δy, Δz)::NTuple{3,FT},
        # xmax::Int, ymax::Int, zmax::Int,

  # Problem configuration
  config = CLIMA.AtmosLESConfiguration("KinematicModel",
                                       N, resolution, xmax, ymax, zmax,
                                       init_saturation_adjustment!,
                                       solver_type=ode_solver,
                                       model=model)
  return config
end

function main()
    CLIMA.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 2500
    ymax = 10
    zmax = 2500

    # Simulation time
    t0 = FT(0)
    # timeend = FT(30 * 60)
    timeend = FT(1)
    # Courant number
    CFL = FT(0.8)

    driver_config = config_saturation_adjustment(FT, N, resolution, xmax, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, init_on_cpu=true, Courant_number=CFL)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)

    @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
