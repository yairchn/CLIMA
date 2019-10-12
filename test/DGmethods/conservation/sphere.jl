#=
Here we solve the equation:
```math
 q + dot(∇, uq) = 0
 p - dot(∇, up) = 0
```
on a sphere to test the conservation of the numerics

The boundary conditions are `p = q` when `dot(n, u) > 0` and
`q = p` when `dot(n, u) < 0` (i.e., `p` flows into `q` and vice-sersa).
=#

using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using Random

using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux_nondiffusive!, flux_diffusive!,
                        source!, boundary_state!,
                        init_aux!, init_state!,
                        init_ode_state, LocalGeometry
import CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                        numerical_flux_nondiffusive!,
                                        numerical_boundary_flux_nondiffusive! 

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array, )
end

struct SphereConservation <: BalanceLaw
end

vars_aux(::SphereConservation,T) = @vars(vel::SVector{3,T})
vars_state(::SphereConservation, T) = @vars(q::T, p::T)

vars_gradient(::SphereConservation, T) = @vars()
vars_diffusive(::SphereConservation, T) = @vars()

flux_diffusive!(::SphereConservation, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing

function init_aux!(::SphereConservation, aux::Vars, g::LocalGeometry)
  x,y,z = g.coord
  r = x^2 + y^2 + z^2
  aux.vel = SVector(cos(10*π*x) * sin(10*π*y) + cos(20 * π * z),
                    exp(sin(π*r)),
                    sin(π * (x + y + z)))
end

function init_state!(::SphereConservation, state::Vars, aux::Vars, (x1,x2,x3), t)
  state.q = rand()
  state.p = rand()
end

function flux_nondiffusive!(::SphereConservation, flux::Grad, state::Vars, auxstate::Vars, t::Real)
  vel = auxstate.vel
  flux.q =  state.q .* vel
  flux.p = -state.p .* vel
end

source!(::SphereConservation, source::Vars, state::Vars, aux::Vars, t::Real) = nothing

struct SphereConservationNumFlux <: NumericalFluxNonDiffusive end

boundary_state!(::CentralNumericalFluxDiffusive, ::SphereConservation, _...) = nothing

function numerical_flux_nondiffusive!(::SphereConservationNumFlux,
                                      bl::BalanceLaw, F::MArray, nM,
                                      QM, auxM, QP, auxP, t)
  FT = eltype(F)
  stateM = Vars{vars_state(bl,FT)}(QM)
  auxstateM = Vars{vars_aux(bl,FT)}(auxM)
  stateP = Vars{vars_state(bl,FT)}(QP)
  auxstateP = Vars{vars_aux(bl,FT)}(auxP)

  unM = dot(nM, auxstateM.vel)
  unP = dot(nM, auxstateP.vel)
  un = (unP + unM) / 2

  Fn = Vars{vars_state(bl,FT)}(F)

  if un > 0
    Fn.q =  un*stateM.q
    Fn.p = -un*stateP.p
  else
    Fn.q =  un*stateP.q
    Fn.p = -un*stateM.p
  end
end

function numerical_boundary_flux_nondiffusive!(nf::SphereConservationNumFlux,
                                               bl::BalanceLaw,
                                               F::MArray{Tuple{nstate}}, nM, QM,
                                               auxM, QP, auxP, bctype, t,
                                               Q1, aux1) where {nstate}
  FT = eltype(F)
  stateM = Vars{vars_state(bl,FT)}(QM)
  auxstateM = Vars{vars_aux(bl,FT)}(auxM)

  un = dot(nM, auxstateM.vel)
  Fn = Vars{vars_state(bl,FT)}(F)

  if un > 0
    Fn.q =  un*stateM.q
    Fn.p = -un*stateM.q
  else
    Fn.q =  un*stateM.p
    Fn.p = -un*stateM.p
  end
end

function run(mpicomm, ArrayType, N, Nhorz, Rrange, timeend, FT, dt)

  topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp
                                         )
  dg = DGModel(SphereConservation(),
               grid,
               SphereConservationNumFlux(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  sum0 = weightedsum(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  sum(Q₀) = %.16e""" eng0 sum0

  max_mass_loss = FT(0)
  max_mass_gain = FT(0)
  cbmass = GenericCallbacks.EveryXSimulationSteps(1) do
    cbsum = weightedsum(Q)
    max_mass_loss = max(max_mass_loss, sum0 - cbsum)
    max_mass_gain = max(max_mass_gain, cbsum - sum0)
    nothing
  end
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbmass,))

  # Print some end of the simulation information
  engf = norm(Q)
  sumf = weightedsum(Q)
  @info @sprintf """Finished
  norm(Q)            = %.16e
  norm(Q) / norm(Q₀) = %.16e
  norm(Q) - norm(Q₀) = %.16e
  max mass loss      = %.16e
  max mass gain      = %.16e
  initial mass       = %.16e
  """ engf engf/eng0 engf-eng0 max_mass_loss max_mass_gain sum0
  max(max_mass_loss, max_mass_gain) / sum0
end

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

  dt = 1e-4
  timeend = 100*dt

  polynomialorder = 4

  Nhorz = 4
  Rrange = 1.0:0.25:2.0

  dim = 3
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64,) #Float32)
      Random.seed!(0)
      @info (ArrayType, FT, dim)
      delta_mass = run(mpicomm, ArrayType, polynomialorder, Nhorz, Rrange, timeend, FT, dt)
      @test abs(delta_mass) < 1e-15
    end
  end
end

nothing
