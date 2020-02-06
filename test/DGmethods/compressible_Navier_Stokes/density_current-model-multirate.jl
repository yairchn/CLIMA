
# Load Packages
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.MultirateRungeKuttaMethod: MultirateRungeKutta
using CLIMA.LowStorageRungeKuttaMethod: LSRK144NiegemannDiehlBusch
using CLIMA.StrongStabilityPreservingRungeKuttaMethod: SSPRK33ShuOsher
using CLIMA.SubgridScaleParameters
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK
using Random
using CLIMA.Atmos: vars_state, vars_aux

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- #
const dim               = 3
const (xmin, xmax)      = (0,12800)
const (ymin, ymax)      = (0,400)
const (zmin, zmax)      = (0,6400)
const Ne                = (32,2,100)
const polynomialorder   = 4
const dt                = 0.1
const timeend           = 300.0

# ------------- Initial condition function ----------- #
"""
@article{doi:10.1002/fld.1650170103,
author = {Straka, J. M. and Wilhelmson, Robert B. and Wicker, Louis J. and Anderson, John R. and Droegemeier, Kelvin K.},
title = {Numerical solutions of a non-linear density current: A benchmark solution and comparisons},
journal = {International Journal for Numerical Methods in Fluids},
volume = {17},
number = {1},
pages = {1-22},
doi = {10.1002/fld.1650170103},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650170103},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.1650170103},
year = {1993}
}
"""
function Initialise_Density_Current!(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT                = eltype(state)
  R_gas::FT         = R_d
  c_p::FT           = cp_d
  c_v::FT           = cv_d
  p0::FT            = MSLP
  # initialise with dry domain
  q_tot::FT         = 0
  q_liq::FT         = 0
  q_ice::FT         = 0
  # perturbation parameters for rising bubble
  rx                = 4000
  rz                = 2000
  xc                = 0
  zc                = 3000
  r                 = sqrt((x1 - xc)^2/rx^2 + (x3 - zc)^2/rz^2)
  θ_ref::FT         = 300
  θ_c::FT           = -15
  Δθ::FT            = 0
  if r <= 1
    Δθ = θ_c * (1 + cospi(r))/2
  end
  qvar              = PhasePartition(q_tot)
  θ                 = θ_ref + Δθ # potential temperature
  π_exner           = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ                 = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density

  P                 = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T                 = P / (ρ * R_gas) # temperature
  U, V, W           = FT(0) , FT(0) , FT(0)  # momentum components
  # energy definitions
  e_kin             = (U^2 + V^2 + W^2) / (2*ρ)/ ρ
  e_pot             = grav * x3
  e_int             = internal_energy(T, qvar)
  E                 = ρ * (e_int + e_kin + e_pot)  #* total_energy(e_kin, e_pot, T, q_tot, q_liq, q_ice)
  state.ρ      = ρ
  state.ρu     = SVector(U, V, W)
  state.ρe     = E
  state.moisture.ρq_tot = FT(0)
end
# --------------- Driver definition ------------------ #
function run(mpicomm, ArrayType,
             topl, dim, Ne, polynomialorder,
             timeend, FT, dt)
  # -------------- Define grid ----------------------------------- #
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- #
  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(300)), FT(0)),
                     AnisoMinDiss{FT}(1),
                     EquilMoist(),
                     NoPrecipitation(),
                     NoRadiation(),
                     NoSubsidence{FT}(),
                     Gravity(), NoFluxBC(), Initialise_Density_Current!)
  # -------------- Define dgbalancelaw --------------------------- #
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  fast_model = AtmosAcousticLinearModel(model)
  slow_model = RemainderModel(model, (fast_model,))

  fast_dg = DGModel(fast_model,
                    grid,
                    Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient();
                    auxstate=dg.auxstate)

  slow_dg = DGModel(slow_model,
                    grid,
                    Rusanov(),
                    CentralNumericalFluxDiffusive(),
                    CentralNumericalFluxGradient();
                    auxstate=dg.auxstate)

  # determine the slow time step
  # elementsize = min_node_distance(grid)
  # slow_dt = 8 * elementsize / soundspeed_air(303.0) / polynomialorder ^ 2
  # nsteps = ceil(Int, timeend / slow_dt)
  slow_dt = dt

  # arbitrary and not needed for stabilty, just for testing
  fast_dt = slow_dt / 10

  Q = init_ode_state(dg, FT(0))

  slow_ode_solver = LSRK144NiegemannDiehlBusch(slow_dg, Q; dt = slow_dt)
  fast_ode_solver = SSPRK33ShuOsher(fast_dg, Q; dt = fast_dt)
  ode_solver = MultirateRungeKutta((slow_ode_solver, fast_ode_solver))

  Δx = xmax/Ne[1]
  Δz = zmax/Ne[3]

  @info @sprintf """Starting density current simulation:
  Time-integrator = MultirateRungeKutta
  Fast solver     = SSPRK33ShuOsher
  Slow solver     = LSRK144NiegemannDiehlBusch
  ArrayType       = %s
  FloatType       = %s
  Δx              = %s
  Δz              = %s
  Δx / Δz         = %s
  Slow Δt         = %s
  Fast Δt         = %s
  Time end        = %s""" ArrayType FT Δx Δz Δx/Δz slow_dt fast_dt timeend

  starttime = Ref(now())

  solve!(Q, ode_solver; timeend=timeend)
  @info @sprintf """Finished at: %s
  """ Dates.format(convert(Dates.DateTime,
                           Dates.now()-starttime[]),
                   Dates.dateformat"HH:MM:SS")
end
# --------------- Test block / Loggers ------------------ #
using Test
let
  CLIMA.init()
  ArrayType = CLIMA.array_type()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  for FT in (Float64,)
    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))

    run(mpicomm, ArrayType,
        topl, dim, Ne, polynomialorder,
        timeend, FT, dt)
  end
end
