using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates


function init_heldsuarez!(bl, state, aux, coords, t)
  FT = eltype(state)
  
  # Set up an initial state
  T_sfc        = FT(300.0)
  temp_profile = IsothermalProfile(T_sfc)
  T, p         = temp_profile(bl.orientation, aux) # pressure is hydrostatic     

  # Add randomness
  #rnd          = FT(1.0 + rand(Uniform(-1e-6, 1e-6)))
  #T           *= rnd 

  # Calculate the initial state variables 
  thermo_state = PhaseDry_given_pT(p, T)
  ρ            = air_density(thermo_state)
  e_int        = internal_energy(thermo_state)
  e_pot        = gravitational_potential(bl.orientation, aux)

  # Set initial state with random perturbation 
  state.ρ      = air_density(T, p)
  state.ρu     = SVector{3, FT}(0, 0, 0)
  state.ρe     = state.ρ * (e_int + e_pot)

  nothing
end

function config_heldsuarez(FT, poly_order, resolution)
  exp_name          = "HeldSuarez"
  
  # Parameters
  domain_height::FT = FT(30e3)
  Rh_ref::FT        = 0
  turb_visc::FT     = 0 # no visc. here

  # Set up a reference state for linearization
  T_sfc             = FT(300.0)
  temp_profile      = IsothermalProfile(T_sfc)
  ref_state         = HydrostaticState(temp_profile_ref, Rh_ref)

  # Set up the atmosphere model
  model = AtmosModel{FT}(
    AtmosGCMConfigType;
                 
    ref_state   = ref_state,
                 
    turbulence  = ConstantViscosityWithDivergence(turb_visc),
    moisture    = DryModel(),
    source      = (Gravity(),),
    init_state  = init_heldsuarez!
  )
  
  config = CLIMA.AtmosGCMConfiguration(
    exp_name, 
    poly_order, 
    resolution,
    domain_height,
    init_heldsuarez!;
    
    model = model
  )

  return config
end


function main()
  CLIMA.init()

  # Driver configuration parameters
  FT            = Float64           # floating type precision
  poly_order    = 4                 # discontinuous Galerkin polynomial order
  n_horz        = 5                 # horizontal element number  
  n_vert        = 5                 # vertical element number
  days          = 100               # experiment day number
  timestart     = FT(0)             # start time (seconds)
  timeend       = FT(days*24*60*60) # end time (seconds)

  # Set up driver configuration
  driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))

  # Set up ODE solver configuration
  ode_solver_type = CLIMA.IMEXSolverType(
    linear_solver = ManyColumnLU,
    solver_method = ARK2GiraldoKellyConstantinescu
  )

  # Set up experiment
  solver_config = CLIMA.setup_solver(
    timestart,
    timeend,
    driver_config,
    ode_solver_type=ode_solver_type,
    Courant_number=0.4,
    init_on_cpu=true,
    CFL_direction=HorizontalDirection()
  )

  # Set up user-defined callbacks
  # TODO: This callback needs to live somewhere else
  #filterorder = 14
  #filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
  #cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
  #    Filters.apply!(
  #      solver_config.Q,
  #      1:size(solver_config.Q, 2),
  #      solver_config.dg.grid,
  #      filter
  #    )
  #    nothing
  #end

  # Run the model
  result = CLIMA.invoke!(
    solver_config;
    #user_callbacks = (cbfilter,),
    check_euclidean_distance = true
  )
end

main()
