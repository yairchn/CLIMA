using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates


"""
    HeldSuarezProfile{F} <: TemperatureProfile
"""
struct HeldSuarezDataConfig{FT}
  temp_profile::HeldSuarezProfile{FT}
end


"""
    HeldSuarezProfile{F} <: TemperatureProfile
    
A latitudinally varying, vertically varying, zonally constant 
temperature profile based on  Held & Suarez (1994).
"""
struct HeldSuarezProfile{FT} <: TemperatureProfile
  p_ref::FT # surface reference pressure of relaxation profile
  T_ref::FT # reference temperature for scale height calculations
  T_min::FT # minium relaxation temperature
  T_equ::FT # surface equatorial temperature
  ΔT_y::FT  # meridional surface temperature gradient
  ΔΘ_z::FT  # vertical potential temperature gradient
end

function (profile::HeldSuarezProfile)(orientation::Orientation, aux::Vars)
  # Parameters
  p_ref = profile.p_ref
  T_ref = profile.T_ref
  T_min = profile.T_min
  T_equ = profile.T_equ
  ΔT_y  = profile.ΔT_y
  Δθ_z  = profile.ΔΘ_z
  σ_b   = profile.σ_b
  
  # Extract relevant coordinate axes
  ϕ     = latitude(orientation, aux)
  z     = altitude(orientation, aux)

  # Height & pressure-related definitions
  H     = R_d * T_ref / grav # scale height
  σ     = exp(-z / H)
 
  # Temperature-related definitions
  T     = T_equ - ΔT_y * sin(ϕ)^2 - Δθ_z * log(σ) * cos(ϕ)^2
  T    *= σ^(R_d / cp_d)
  T     = max(T_min, T)
  p     = p_ref * σ

  return T, p
end


"""
    init_heldsuarez!(bl, state, aux, coords, t)    
"""
function init_heldsuarez!(bl, state, aux, coords, t)
  FT           = eltype(state)
  
  # Add random perturbation to temperature profile 
  T, p         = bl.data_config.temp_profile(bl.orientation, aux)
  T           *= FT(1.0 + rand(Uniform(-1e-6, 1e-6)))
  
  # Configure thermodynamic state
  thermo_state = PhaseDry_given_pT(p, T)
  ρ            = air_density(thermo_state)
  e_int        = internal_energy(thermo_state)
  e_pot        = gravitational_potential(bl.orientation, aux)

  # Set initial state with random perturbation 
  state.ρ      = ρ
  state.ρu     = SVector{3, FT}(0, 0, 0)
  state.ρe     = state.ρ * (e_int + e_pot)

  nothing
end


"""
    held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
"""
function held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
  FT = eltype(state)

  # Extract relevant axes
  ϕ      = latitude(orientation, aux)
  
  # Extract the current state
  ρ      = state.ρ
  ρu     = state.ρu
  ρe     = state.ρe

  # Extract relevant themodynamic variables
  e_int  = internal_energy(bl.moisture, bl.orientation, state, aux)
  T      = air_temperature(e_int)
  
  # Extract reference state temperature profile
  p_ref  = bl.data_config.temp_profile.p_ref
  T_relax, ~ = bl.ref_state.temp_profile(bl.orientation, aux)
   
  # Calculate Held-Suarez relaxation coefficients
  k_f    = FT(1 / day)
  k_s    = FT(1 / 4 / day)
  k_a    = FT(1 / 40 / day)
  σ_b    = FT(0.7)
  p      = air_pressure(T, ρ)
  σ      = p / p_ref
  Δσ     = (σ - σ_b) / (1 - σ_b)
  σ_cut  = max(0, Δσ)
  k_T    = k_a + (k_s - k_a) * cos(ϕ)^4 * σ_cut
  k_v    = k_f * σ_cut

  # Apply Held-Suarez forcing
  source.ρu -= k_v * projection_tangential(bl.orientation, aux, ρu)
  source.ρe -= k_T * ρ * cv_d * (T - T_relax)
end


"""
    config_heldsuarez(FT, poly_order, resolution)
"""
function config_heldsuarez(FT, poly_order, resolution)
  exp_name          = "HeldSuarez"
  
  domain_height     = FT(30e3)
  temp_profile_hs   = HeldSuarezProfile{FT}(
                          p_ref = MLSP,
                          T_ref = 255,
                          T_min = 200,
                          T_equ = 315,
                          ΔT_y  = 60, 
                          ΔΘ_z  = 10
                      ) 

  # Set up a dry reference state for linearization
  temp_profile_ref  = LinearTemperatureProfile(T_min, T_sfc, Γ)
  ref_state         = HydrostaticState(temp_profile_ref, FT(0))

  # Rayleigh sponge to dampen flow at the top of the domain 
  z_sponge          = FT(15e3) # height at which sponge begins
  α_relax           = FT(1/60.0/30.0) # sponge relaxation rate in (1/seconds) 
  u_relax           = SVector(FT(0), FT(0), FT(0)) # relaxation velocity
  exp_sponge        = 2 # sponge exponent for squared-sinusoid profile
  sponge            = RayleighSponge{FT}(
                        domain_height, 
                        z_sponge, 
                        α_relax, 
                        u_relax, 
                        exp_sponge
                      )

  # Viscous sponge to dampen flow at the top of the domain
  dyn_visc_bg       = FT(0.0) 
  z_sponge          = FT(15e3)
  dyn_visc_sp       = FT(1e7)
  exp_sponge        = FT(2)
  turb_model        = ConstantViscosityWithDivergence(dyn_visc_bg), 
  #turb_model        = ConstantViscousSponge(
  #                      dyn_visc_bg, 
  #                      domain_height, 
  #                      z_sponge, 
  #                      dyn_visc_sp,
  #                      exp_sponge
  #                    )

  # Set up the atmosphere model
  model = AtmosModel{FT}(
    AtmosGCMConfiguration;
                 
    ref_state   = ref_state,
                 
    turbulence  = turb_model,  
    moisture    = DryModel(),
    source      = (Gravity(), Coriolis(), held_suarez_forcing!, sponge),
    init_state  = init_heldsuarez!, 
    data_config = HeldSuarezDataConfig(
                    temp_profile_hs
                  )
  )
  
  config = CLIMA.Atmos_GCM_Configuration(
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
  FT            = Float32           # floating type precision
  poly_order    = 5                 # discontinuous Galerkin polynomial order
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
    Courant_number=0.1,
    forcecpu=true, 
    diffdir=EveryDirection()
  )

  # Set up user-defined callbacks
  # TODO: This callback needs to live somewhere else 
  filterorder = 8
  filter = ExponentialFilter(
    solver_config.dg.grid, 
    0, 
    filterorder,
    -log(eps(FT))
  )
  cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
      Filters.apply!(
        solver_config.Q, 
        1:size(solver_config.Q, 2),
        solver_config.dg.grid, 
        filter
      )
      nothing
  end

  # Run the model
  result = CLIMA.invoke!(
    solver_config;
    user_callbacks = (cbfilter,),
    check_euclidean_distance = true
  )
end

main()
