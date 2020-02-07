# Load Packages
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods: courant
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.LinearSolvers
using CLIMA.ColumnwiseLUSolver
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
using CLIMA.Atmos: vars_state, vars_aux, soundspeed

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- #
const (xmin,xmax)      = (0,1000)
const (ymin,ymax)      = (0,400)
const (zmin,zmax)      = (0,1000)
const Ne        = (20,2,100)
const polynomialorder = 4
const dim       = 3
const dt        = 0.05
const timeend   = 100.0

# ------------- Initial condition function ----------- #
"""
@article{doi:10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2,
author = {Robert, A},
title = {Bubble Convection Experiments with a Semi-implicit Formulation of the Euler Equations},
journal = {Journal of the Atmospheric Sciences},
volume = {50},
number = {13},
pages = {1865-1873},
year = {1993},
doi = {10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
URL = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
eprint = {https://doi.org/10.1175/1520-0469(1993)050<1865:BCEWAS>2.0.CO;2},
}
"""
function Initialise_Rising_Bubble!(state::Vars, aux::Vars, (x1,x2,x3), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP

  xc::FT        = 500
  zc::FT        = 260
  r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  rc::FT        = 250
  θ_ref::FT     = 303
  Δθ::FT        = 0

  if r <= rc
    Δθ          = FT(1//2)
  end
  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
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
                     HydrostaticState(DryAdiabaticProfile(typemin(FT), FT(303)), FT(0)),
                     Vreman{FT}(C_smag),
                     EquilMoist(),
                     NoPrecipitation(),
                     NoRadiation(),
                     NoSubsidence{FT}(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)

  # -------------- Define dgbalancelaw --------------------------- #
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

  linmodel = AtmosAcousticGravityLinearModel(model)
  lindg = DGModel(linmodel,
                  grid,
                  Rusanov(),
                  CentralNumericalFluxDiffusive(),
                  CentralNumericalFluxGradient();
                  direction=VerticalDirection(),
                  auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0))

  linearsolver = ManyColumnLU()
  ark = ARK548L2SA2KennedyCarpenter(dg, lindg, linearsolver, Q; dt = dt, t0 = 0)

  Δx = xmax/Ne[1]
  Δz = zmax/Ne[3]

  @info @sprintf """Starting rising bubble simulation:
  Time-integrator = ARK548L2SA2KennedyCarpenter
  Linear solver   = ManyColumnLU
  ArrayType       = %s
  FloatType       = %s
  Δx              = %s
  Δz              = %s
  Δx / Δz         = %s
  Δt              = %s
  Time end        = %s""" ArrayType FT Δx Δz Δx/Δz dt timeend

  starttime = Ref(now())

  solve!(Q, ark; timeend=timeend, adjustfinalstep=false)

  @info @sprintf """Finished at: %s
  """ Dates.format(convert(Dates.DateTime,
                           Dates.now()-starttime[]),
                   Dates.dateformat"HH:MM:SS")

  function local_courant(m::AtmosModel, state::Vars, aux::Vars, diffusive::Vars, Δx)
    u = state.ρu/state.ρ
    return dt * (norm(u) + soundspeed(m.moisture, m.orientation, state, aux)) / Δx
  end

  c = courant(local_courant, dg, model, Q, EveryDirection())
  c_h = courant(local_courant, dg, model, Q, HorizontalDirection())
  c_v = courant(local_courant, dg, model, Q, VerticalDirection())
  @info @sprintf("""Courant numbers:
  c   = %.2e
  c_h = %.2e
  c_v = %.2e
  """, c, c_h, c_v)
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
  brickrange = (range(Float64(xmin); length=Ne[1]+1, stop=xmax),
                range(Float64(ymin); length=Ne[2]+1, stop=ymax),
                range(Float64(zmin); length=Ne[3]+1, stop=zmax))
  topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
  run(mpicomm, ArrayType,
      topl, dim, Ne, polynomialorder,
      timeend, Float64, dt)
end
