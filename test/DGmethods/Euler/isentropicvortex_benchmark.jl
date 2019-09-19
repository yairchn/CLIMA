using CLIMA: haspkg
using CLIMA.Mesh.Topologies: BrickTopology
using CLIMA.Mesh.Grids: DiscontinuousSpectralElementGrid
using CLIMA.DGmethods: DGModel, init_ode_param, init_ode_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralGradPenalty,
                                       CentralNumericalFluxDiffusive
using CLIMA.ODESolvers: solve!, gettime
using CLIMA.LowStorageRungeKuttaMethod: LSRK54CarpenterKennedy
using CLIMA.DifferentialEquationsMethods: DifferentialEquationsMethod
using DifferentialEquations: CarpenterKennedy2N54
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MPIStateArrays: euclidean_distance
using CLIMA.PlanetParameters: kappa_d
using CLIMA.MoistThermodynamics: air_density, total_energy, soundspeed_air
using CLIMA.Atmos: AtmosModel, NoOrientation, NoReferenceState,
                   DryModel, NoRadiation, PeriodicBC,
                   NoViscosity, vars_state
using CLIMA.VariableTemplates: flattenednames

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

const use_DifferentialEquations_lsrk = true
const numsteps = 1
const dims = 3
const DFloat = Float64
# those two parameters collectively determine the size of the problem
const polynomialorder = 4
const numelems = ntuple(_ -> 10, dims)

function main()
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = Dict("DEBUG" => Logging.Debug,
                  "WARN"  => Logging.Warn,
                  "ERROR" => Logging.Error,
                  "INFO"  => Logging.Info)[ll]

  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  

  @info @sprintf """Configuration
                    ArrayType = %s
                    DFloat    = %s
                    dims      = %d
                    """ "$ArrayType" "$DFloat" dims

  setup = IsentropicVortexSetup{DFloat}()

  run(mpicomm, setup, ArrayType, 1)
end

function run(mpicomm, setup, ArrayType, level)
  brickrange = ntuple(dims) do dim
    range(-setup.domain_halflength; length=numelems[dim] + 1, stop=setup.domain_halflength)
  end

  topology = BrickTopology(mpicomm,
                           brickrange;
                           periodicity=ntuple(_ -> true, dims))

  grid = DiscontinuousSpectralElementGrid(topology,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder)

  initialcondition! = function(args...)
    isentropicvortex_initialcondition!(setup, args...)
  end
  model = AtmosModel(NoOrientation(),
                     NoReferenceState(),
                     NoViscosity(),
                     DryModel(),
                     NoRadiation(),
                     nothing,
                     PeriodicBC(),
                     initialcondition!)

  dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralGradPenalty())
  param = init_ode_param(dg)


  # determine the time step
  elementsize = minimum(step.(brickrange))
  dt = elementsize / soundspeed_air(setup.T∞) / polynomialorder ^ 2
  
  timeend = numsteps * dt

  Q = init_ode_state(dg, param, DFloat(0))
 
  if use_DifferentialEquations_lsrk
    lsrk = DifferentialEquationsMethod(CarpenterKennedy2N54(williamson_condition=false), dg, Q; dt = dt, t0 = 0)
  else
    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
  end

  eng0 = norm(Q)
  numelems_print = dims == 2 ? (numelems..., 0) : numelems
  @info @sprintf """Starting refinement level %d
                    numelems  = (%d, %d, %d)
                    dt        = %.16e
                    norm(Q₀)  = %.16e
                    """ level numelems_print... dt eng0
  
  solve!(Q, lsrk, param; timeend=timeend)

  # final statistics
  Qe = init_ode_state(dg, param, timeend)
  engf = norm(Q)
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished refinement level %d
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ level engf engf/eng0 engf-eng0 errf errf/engfe
  errf
end

Base.@kwdef struct IsentropicVortexSetup{DFloat}
  p∞::DFloat = 10 ^ 5
  T∞::DFloat = 300
  ρ∞::DFloat = air_density(DFloat(T∞), DFloat(p∞))
  translation_speed::DFloat = 150
  translation_angle::DFloat = pi / 4
  vortex_speed::DFloat = 50
  vortex_radius::DFloat = 1 // 200
  domain_halflength::DFloat = 1 // 20
end

function isentropicvortex_initialcondition!(setup, state, aux, coords, t)
  DFloat = eltype(state)
  x = MVector(coords)

  ρ∞ = setup.ρ∞
  p∞ = setup.p∞
  T∞ = setup.T∞
  translation_speed = setup.translation_speed
  α = setup.translation_angle
  vortex_speed = setup.vortex_speed
  R = setup.vortex_radius
  L = setup.domain_halflength

  u∞ = SVector(translation_speed * cos(α), translation_speed * sin(α), 0)

  x .-= u∞ * t
  # make the function periodic
  x .-= floor.((x + L) / 2L) * 2L

  @inbounds begin
    r = sqrt(x[1] ^ 2 + x[2] ^ 2)
    δu_x = -vortex_speed * x[2] / R * exp(-(r / R) ^ 2 / 2)
    δu_y =  vortex_speed * x[1] / R * exp(-(r / R) ^ 2 / 2)
  end
  u = u∞ .+ SVector(δu_x, δu_y, 0)

  T = T∞ * (1 - kappa_d * vortex_speed ^ 2 / 2 * ρ∞ / p∞ * exp(-(r / R) ^ 2))
  # adiabatic/isentropic relation
  p = p∞ * (T / T∞) ^ (DFloat(1) / kappa_d)
  ρ = air_density(T, p)

  state.ρ = ρ
  state.ρu = ρ * u
  e_kin = u' * u / 2
  state.ρe = ρ * total_energy(e_kin, DFloat(0), T)
end

main()
