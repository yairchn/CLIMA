# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.LinearSolvers
using CLIMA.GeneralizedMinimalResidualSolver
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
using CLIMA.Atmos: vars_state, ReferenceState
import CLIMA.Atmos: atmos_init_aux!, vars_aux

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array,)
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- # 
const polynomialorder = 4
const dim       = 3
const domain_start = (0, 0, 0)
const domain_end = (1000, dim == 2 ? 100 : 1000, 1000)
const Ne = (10, dim == 2 ? 1 : 10, 10)
const Δxyz = @. (domain_end - domain_start) / Ne / polynomialorder
const dt_factor = 20
const dt = dt_factor * min(Δxyz...) / soundspeed_air(300.0) / polynomialorder
const timeend   = 10dt
const smooth_bubble = true
const dry = true
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
  if dim == 2
    r             = sqrt((x1 - xc)^2 + (x3 - zc)^2)
  else
    r             = sqrt((x1 - xc)^2 + (x2 - xc)^2 + (x3 - zc)^2)
  end
  rc::FT        = 250
  θ_ref::FT     = 303
  Δθ::FT        = 0
  θ_c::FT       = 1 // 2
  
  if smooth_bubble
    a::FT   =  50
    s::FT   = 100
    if r <= a
      Δθ = θ_c
    elseif r > a
      Δθ = θ_c * exp(-(r - a)^2 / s^2)
    end
  else
    if r <= rc
      Δθ          = θ_c
    end
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
  if !dry
    state.moisture.ρq_tot = FT(0)
  end
end

struct RisingBubbleReferenceState <: ReferenceState end
vars_aux(::RisingBubbleReferenceState, DT) = @vars(ρ::DT, p::DT, T::DT, ρe::DT)
function atmos_init_aux!(m::RisingBubbleReferenceState, atmos::AtmosModel, aux::Vars, geom::LocalGeometry)
  x1, x2, x3 = geom.coord
  FT            = eltype(aux)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP
  θ_ref::FT     = 303

  θ            = θ_ref
  π_exner      = FT(1) - grav / (c_p * θ) * x3 # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  e_kin        = FT(0)
  e_pot        = grav * x3
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)

  aux.ref_state.ρ = ρ
  aux.ref_state.ρe = ρe_tot
  aux.ref_state.p = P
  aux.ref_state.T = T
end

# --------------- Driver definition ------------------ # 
function run(mpicomm, ArrayType, LinearType,
             topl, dim, Ne, polynomialorder, 
             timeend, FT, dt)
  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                           )
  # -------------- Define model ---------------------------------- # 
  moistmodel = dry ? DryModel() : EquilMoist()
  model = AtmosModel(FlatOrientation(),
                     RisingBubbleReferenceState(),
                     Vreman{FT}(C_smag),
                     moistmodel,
                     NoRadiation(),
                     Gravity(),
                     NoFluxBC(),
                     Initialise_Rising_Bubble!)
  # -------------- Define dgbalancelaw --------------------------- # 
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  linmodel = LinearType(model)
  lindg = DGModel(linmodel,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty(); auxstate=dg.auxstate)

  Q = init_ode_state(dg, FT(0))

  linearsolver = GeneralizedMinimalResidual(10, Q, sqrt(eps(FT)))
  ark = ARK548L2SA2KennedyCarpenter(dg, lindg, linearsolver, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  ArrayType = %s
  FloatType = %s""" eng0 ArrayType FT

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(ark),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  solve!(Q, ark; timeend=timeend, callbacks=(cbinfo,))
  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
engf/eng0
end
# --------------- Test block / Loggers ------------------ # 
using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    FloatType = (Float64,)
    for FT in FloatType
      brickrange = ntuple(d -> range(FT(domain_start[d]); length=Ne[d]+1, stop=domain_end[d]), 3)
      periodicity = (false, dim == 2 ? true : false, false)
      topl = StackedBrickTopology(mpicomm, brickrange, periodicity = periodicity)
      for LinearType in (AtmosAcousticLinearModel,)
        engf_eng0 = run(mpicomm, ArrayType, LinearType,
                        topl, dim, Ne, polynomialorder, 
                        timeend, FT, dt)
      end
    end
  end
end

#nothing
