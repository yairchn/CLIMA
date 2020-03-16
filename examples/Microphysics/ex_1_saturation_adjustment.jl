using Dates
using Distributions
using DocStringExtensions
using LinearAlgebra
using Logging
using MPI
using Printf
using Random
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Grids
using CLIMA.GenericCallbacks
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Topologies
using CLIMA.MoistThermodynamics
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using CLIMA.VTK

using CLIMA.Mesh.Grids: VerticalDirection,
                        HorizontalDirection,
                        min_node_distance

import CLIMA.DGmethods: BalanceLaw,
                        vars_gradient,
                        vars_diffusive,
                        flux_nondiffusive!,
                        flux_diffusive!,
                        source!,
                        wavespeed,
                        vars_aux,
                        vars_state,
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
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk,
#         and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}
# ------------------------ Description ------------------------- #

Base.@kwdef struct KinematicBC{M, E, Q}
    momentum::M = Impenetrable(FreeSlip())
    energy::E = Insulating()
    moisture::Q = Impermeable()
end

struct KinematicModelConfig{FT}
  xmax::FT
  ymax::FT
  zmax::FT
  wmax::FT
end

struct KinematicModel{FT,O,M,P,S,BC,IS,DC} <: BalanceLaw
  orientation::O
  moisture::M
  precipitation::P
  source::S
  boundarycondition::BC
  init_state::IS
  data_config::DC
end

function KinematicModel{FT}(::Type{AtmosLESConfigType};
                         orientation::O=FlatOrientation(),
                         moisture::M=EquilMoist{FT}(),
                         precipitation::P=NoPrecipitation(),
                         source::S=nothing,
                         boundarycondition::BC=NoFluxBC(),
                         init_state::IS=nothing,
                         data_config::DC=nothing) where{FT<:AbstractFloat,O,M,P,S,BC,IS,DC}

  @assert init_state ≠ nothing

  atmos = (
        orientation,
        moisture,
        precipitation,
        source,
        boundarycondition,
        init_state,
        data_config,
       )

  return KinematicModel{FT,typeof.(atmos)...}(atmos...)
end


function vars_state(m::KinematicModel, FT)
  @vars begin
    ρ::FT
    ρu::SVector{3,FT}
    ρe::FT
    ρq_tot::FT
  end
end

function vars_aux(m::KinematicModel, FT)
  @vars begin
    q_tot::FT
    q_vap::FT
    q_liq::FT
    q_ice::FT
    T::FT
    p::FT
    x::FT
    z::FT
  end
end

function init_state!(m::KinematicModel, state::Vars, aux::Vars, coords, t, args...)
  m.init_state(m, state, aux, coords, t, args...)
end

function init_kinematic_eddy!(ed, state, aux, (x,y,z), t)
  FT = eltype(state)

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
  dc = ed.data_config

  # velocity as derivative of streamfunction
  ρu::FT = dc.wmax * dc.xmax/dc.zmax * cos(π * z/dc.zmax) * cos(2*π * x/dc.xmax)
  ρw::FT = 2*dc.wmax * sin(π * z/dc.zmax) * sin(2*π * x/dc.xmax)
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

function init_aux!(m::KinematicModel, aux::Vars, geom::LocalGeometry)

  x, y, z = geom.coord

  FT = eltype(aux)
  # initial condition
  θ_0::FT    = 289         # K
  p_0::FT    = 101500      # Pa
  p_1000::FT = 100000      # Pa
  qt_0::FT   = 7.5 * 1e-3  # kg/kg
  z_0::FT    = 0           # m

  R_m, cp_m, cv_m, γ = gas_constants(PhasePartition(qt_0))

  # Pressure profile assuming hydrostatic and constant θ and qt profiles.
  # It is done this way to be consistent with Arabas paper.
  # It's not neccesarily the best way to initialize with our model variables.
  p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
              R_d / cp_d * grav / θ_0 / R_m * (z - z_0)
             )^(cp_d / R_d)

  aux.p = p  # for prescribed pressure gradient (kinematic setup)

  aux.x = x
  aux.z = z

  aux.q_tot = qt_0
  aux.q_vap = FT(0)
  aux.q_liq = FT(0)
  aux.q_ice = FT(0)

end

function boundary_state!(nf, m::KinematicModel, args...)
    bc_kin_boundary_state!(nf, m.boundarycondition, m, args...)
end
@generated function bc_kin_boundary_state!(
    nf,
    tup::Tuple,
    m,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(tup)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> bc_kin_boundary_state!(
                nf,
                tup[i],
                m,
                state⁺,
                aux⁺,
                n,
                state⁻,
                aux⁻,
                bctype,
                t,
                args...,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

function bc_kin_boundary_state!(nf, bc::KinematicBC, m, args...)
    # bc_kin_momentum_boundary_state!(nf, bc.momentum, m, args...)
    # bc_kin_energy_boundary_state!(nf, bc.energy, m, args...)
    # bc_kin_moisture_boundary_state!(nf, bc.moisture, m, args...)
end

function bc_kinematic_eddy!(m::KinematicModel, state⁺::Vars, state⁻::Vars, _...)

   FT = eltype(state⁻)

   state⁺.ρ = state⁻.ρ
   state⁺.ρe = state⁻.ρe_tot
   state⁺.ρq_tot = state⁻.ρq_tot

end

@inline function wavespeed(m::KinematicModel, nM, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  u = ρinv * state.ρu
  return abs(dot(nM, u))
end


# ------------------------------------------------------------------ BOILER PLATE :)
vars_gradient(m::KinematicModel, FT) = @vars()
vars_diffusive(m::KinematicModel, FT) = @vars()

function source!(m::KinematicModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end

@inline function flux_nondiffusive!(m::KinematicModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end

function gradvariables!(m::KinematicModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

function diffusive!(m::KinematicModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
end

@inline function flux_diffusive!(
    m::KinematicModel, flux::Grad, state::Vars, diffusive::Vars,
    hyperdiffusive::Vars, aux::Vars, t::Real)
end

@inline function flux_diffusive!(
    m::KinematicModel, flux::Grad, state::Vars, τ, d_h_tot)
end

function update_aux!(dg::DGModel, m::KinematicModel, Q::MPIStateArray, t::Real)
  return true
end

#function integral_load_aux!(m::KinematicModel, integ::Vars, state::Vars, aux::Vars) end
#function integral_set_aux!(m::KinematicModel, aux::Vars, integ::Vars) end
#function reverse_integral_load_aux!(m::KinematicModel, integ::Vars, state::Vars, aux::Vars) end
#function reverse_integral_set_aux!(m::KinematicModel, aux::Vars, integ::Vars) end
# ------------------------------------------------------------------


function config_kinematic_eddy(FT, N, resolution, xmax, ymax, zmax, wmax)

  # Choose explicit solver
  ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  IS = typeof(init_kinematic_eddy!)
  BC = typeof(bc_kinematic_eddy!)

  kmc = KinematicModelConfig(
      FT(xmax),
      FT(ymax),
      FT(zmax),
      FT(wmax)
)

  # Set up the model
  model = KinematicModel{FT}(AtmosLESConfigType;
      boundarycondition = KinematicBC(),
      init_state = init_kinematic_eddy!,
      data_config = kmc
  )

  config = CLIMA.AtmosLESConfiguration(
      "KinematicModel",
      N,
      resolution,
      FT(xmax),
      FT(ymax),
      FT(zmax),
      init_kinematic_eddy!;
      solver_type = ode_solver,
      model = model
  )

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
    # Eddy max velocity
    wmax = FT(0.6)

    # Simulation time
    t0 = FT(0)
    # timeend = FT(30 * 60)
    timeend = FT(1)
    # Courant number
    CFL = FT(0.8)

    driver_config = config_kinematic_eddy(
        FT, N, resolution, xmax, ymax, zmax, wmax)
    solver_config = CLIMA.setup_solver(
        t0, timeend, driver_config; ode_dt=FT(0.1),
        init_on_cpu=true, Courant_number=CFL
    )

    mpicomm = MPI.COMM_WORLD

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
      if s
        starttime[] = now()
      else
        energy = norm(solver_config.Q)
        @info @sprintf("""Update
                       norm(Q) = %.16e""",
                       energy)
      end
    end

    # output for paraview
    model = driver_config.bl
    step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1)  do (init=false)
      mkpath("vtk/")
      outprefix = @sprintf("vtk/new_ex_1_mpirank%04d_step%04d",
                           MPI.Comm_rank(mpicomm), step[1])
      @info "doing VTK output" outprefix
      writevtk(outprefix,
               solver_config.Q,
               solver_config.dg,
               flattenednames(vars_state(model,FT)),
               solver_config.dg.auxstate,
               flattenednames(vars_aux(model,FT))
      )
      step[1] += 1
      nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter, cbinfo, cbvtk),
                          check_euclidean_distance=true)

    @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
