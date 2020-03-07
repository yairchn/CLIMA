using LinearAlgebra
using StaticArrays
using Test
using MPI
using Printf

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using CLIMA.Courant
using CLIMA.VTK: writevtk, writepvtu

import CLIMA.Atmos: vars_state, vars_aux
import CLIMA.DGmethods: courant, init_ode_state
import CLIMA.Grids: VerticalDirection, HorizontalDirection

const output_vtk = true

Base.@kwdef struct AcousticWaveSetup{FT}
  domain_height::FT = 10e3
  T_ref::FT = 300
  α::FT = 3
  γ::FT = 100
  nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
  # callable to set initial conditions
  FT = eltype(state)

  λ = longitude(bl.orientation, aux)
  φ = latitude(bl.orientation, aux)
  z = altitude(bl.orientation, aux)

  β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
  f = (1 + cos(FT(π) * β)) / 2
  g = sin(setup.nv * FT(π) * z / setup.domain_height)
  Δp = setup.γ * f * g
  p = aux.ref_state.p + Δp

  ts       = PhaseDry_given_pT(p, setup.T_ref)
  q_pt     = PhasePartition(ts)
  e_pot    = gravitational_potential(bl.orientation, aux)
  e_int    = internal_energy(ts)

  state.ρ  = air_density(ts)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (e_int + e_pot)
  nothing
end

function config_acousticwave(FT, N, resolution)

  setup = AcousticWaveSetup{FT}()
  orientation = SphericalOrientation()
  ref_state = HydrostaticState(IsothermalProfile(setup.T_ref), FT(0))
  turbulence = ConstantViscosityWithDivergence(FT(0))
  model = AtmosModel{FT}(AtmosGCMConfiguration;
                         orientation       = orientation,
                         ref_state         = ref_state,
                         turbulence        = turbulence,
                         moisture          = DryModel(),
                         source            = Gravity(),
                         init_state        = setup)

  # max ~ CFL(4)
  # ode_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
  ode_solver = CLIMA.MRRKSolverType(solver_method=MultirateRungeKutta,
                                    slow_method=LSRK144NiegemannDiehlBusch,
                                    # fast_method=LSRK144NiegemannDiehlBusch,
                                    fast_method=LSRK54CarpenterKennedy,
                                    numsubsteps=200,
                                    linear_model=AtmosAcousticGravityLinearModel)

  config = CLIMA.Atmos_GCM_Configuration("AcousticWave", N, resolution,
                                         setup.domain_height,
                                         setup;
                                         solver_type=ode_solver,
                                         model = model)
  return config
end

function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution
    nelem_horz = 4
    nelem_vert = 6
    resolution = (nelem_horz, nelem_vert)

    t0 = FT(0)
    timeend = FT(33 * 60 * 60)
    # Courant number
    CFL = FT(4)
    odedt = FT(200)

    driver_config = config_acousticwave(FT, N, resolution)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config,
                                       ode_dt=odedt)

    # Set up the filter callback
    filterorder = 18
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(solver_config.Q, 1:size(solver_config.Q, 2),
                       solver_config.dg.grid, filter, VerticalDirection())
        nothing
    end

    steps = 0
    perstep = 1
    cbcourantnumbers = GenericCallbacks.EveryXSimulationSteps(perstep) do
          steps += perstep
          dg =  solver_config.dg
          m = dg.balancelaw
          Q = solver_config.Q
          Δt = solver_config.dt
          cfl_v = courant(nondiffusive_courant, dg, m, Q, Δt, VerticalDirection())
          cfl_h = courant(nondiffusive_courant, dg, m, Q, Δt, HorizontalDirection())
          cfla_v = courant(advective_courant, dg, m, Q, Δt, VerticalDirection())
          cfla_h = courant(advective_courant, dg, m, Q, Δt, HorizontalDirection())
          cfld_v = courant(diffusive_courant, dg, m, Q, Δt, VerticalDirection())
          cfld_h = courant(diffusive_courant, dg, m, Q, Δt, HorizontalDirection())

          fΔt = solver_config.solver.fast_solver.dt
          cflin_v = courant(nondiffusive_courant, dg, m, Q, fΔt, VerticalDirection())
          cflin_h = courant(nondiffusive_courant, dg, m, Q, fΔt, HorizontalDirection())

          @info @sprintf """
          ================================
          Courant numbers at step: %s

          Outer (slow) method:
          --------------------------------
          Vertical Acoustic CFL    = %.2g
          Horizontal Acoustic CFL  = %.2g
          --------------------------------
          Vertical Advection CFL   = %.2g
          Horizontal Advection CFL = %.2g
          --------------------------------
          Vertical Diffusion CFL   = %.2g
          Horizontal Diffusion CFL = %.2g
          --------------------------------

          Inner (fast) method:
          --------------------------------
          Vertical Acoustic CFL    = %.2g
          Horizontal Acoustic CFL  = %.2g
          --------------------------------
          ================================
          """  steps cfl_v cfl_h cfla_v cfla_h cfld_v cfld_h cflin_v cflin_h

          return nothing
    end

    callbacks = (cbfilter, cbcourantnumbers)

    if output_vtk
      # create vtk dir
      vtkdir = "vtk_acousticwave" *
        "_poly$(N)_horz$(nelem_horz)_vert$(nelem_vert)_CFL$(CFL)"
      mkpath(vtkdir)

      mpicomm = MPI.COMM_WORLD
      vtkstep = 0
      # output initial step
      do_output(mpicomm, vtkdir, vtkstep,
                solver_config.dg, solver_config.Q,
                driver_config.bl)

      # setup the output callback
      cbvtk = GenericCallbacks.EveryXSimulationSteps(2) do
        vtkstep += 1
        Qe = init_ode_state(solver_config.dg, gettime(solver_config.solver))
        do_output(mpicomm, vtkdir, vtkstep,
                  solver_config.dg, solver_config.Q,
                  driver_config.bl)
      end
      callbacks = (callbacks..., cbvtk)
    end

    result = CLIMA.invoke!(solver_config;
                           user_callbacks=callbacks,
                           check_euclidean_distance=true)
end

function do_output(mpicomm, vtkdir, vtkstep, dg, Q, model, testname = "acousticwave")
  ## name of the file that this MPI rank will write
  filename = @sprintf("%s/%s_mpirank%04d_step%04d",
                      vtkdir, testname, MPI.Comm_rank(mpicomm), vtkstep)

  statenames = flattenednames(vars_state(model, eltype(Q)))
  auxnames = flattenednames(vars_aux(model, eltype(Q)))
  writevtk(filename, Q, dg, statenames, dg.auxstate, auxnames)

  ## Generate the pvtu file for these vtk files
  if MPI.Comm_rank(mpicomm) == 0
    ## name of the pvtu file
    pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

    ## name of each of the ranks vtk files
    prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
      @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
    end

    writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

    @info "Done writing VTK: $pvtuprefix"
  end
end

main()
