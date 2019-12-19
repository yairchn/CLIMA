# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
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

const ArrayType = CLIMA.array_type()

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- # 
const (xmin,xmax)      = (0,1000)
const (ymin,ymax)      = (0,400)
const (zmin,zmax)      = (0,1000)
const Ne        = (10,2,10)
const N = 4
const dim       = 3
const dt        = 0.01
const timeend   = 10
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
function run(mpicomm, 
             topl, dim, Ne, N, 
             timeend, FT, dt)
  # --------------- Testing global_stats -------------# 
  function gather_global_stats(mpicomm, 
                               dg, 
                               Q,
                               dt)

    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)

    # extract grid information
    bl = dg.balancelaw
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get the state, auxiliary and geo variables onto the host if needed
    if Array ∈ typeof(Q).parameters
        localQ    = Q.realdata
        localaux  = dg.auxstate.realdata
        localvgeo = grid.vgeo
        localdiff = dg.diffstate.realdata
    else
        localQ    = Array(Q.realdata)
        localaux  = Array(dg.auxstate.realdata)
        localvgeo = Array(grid.vgeo)
        localdiff = Array(dg.diffstate.realdata)
    end
    FT = eltype(localQ)
    zvals = zeros(Nqk, nvertelem)
    thermoQ = zeros(Nq*Nq*Nqk,1,nrealelem)

    (Σρ, Σρu, Σρv, Σρw, Σρe, Σρq, ΣM) = (zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),zero(FT))
    
    # Accumulate variables across nodes and elements
    for eh in 1:nhorzelem
      for ev in 1:nvertelem
        e = ev + (eh - 1) * nvertelem
        for k in 1:Nqk
          for j in 1:Nq
            for i in 1:Nq
              ijk     = i + Nq * ((j-1) + Nq * (k-1))
              M       = localvgeo[ijk, grid.Mid, e]
              Σρ      += M * localQ[ijk,1,e]
              Σρu     += M * localQ[ijk,2,e]
              Σρv     += M * localQ[ijk,3,e]
              Σρw     += M * localQ[ijk,4,e]
              Σρe     += M * localQ[ijk,5,e]
              Σρq     += M * localQ[ijk,6,e]
              ΣM      += M
            end
          end
        end
      end
    end
    
    Σ_red  = MPI.Reduce(ΣM, +, 0, mpicomm)
    # Collapse DG averages across ranks
    ρ̅      = MPI.Reduce(Σρ, +, 0, mpicomm) / Σ_red
    ρ̅u̅     = MPI.Reduce(Σρu, +, 0, mpicomm) / Σ_red
    ρ̅v̅     = MPI.Reduce(Σρv, +, 0, mpicomm) / Σ_red
    ρ̅w̅     = MPI.Reduce(Σρw, +, 0, mpicomm) / Σ_red
    ρ̅e̅     = MPI.Reduce(Σρe, +, 0, mpicomm) / Σ_red
    ρ̅q̅     = MPI.Reduce(Σρq, +, 0, mpicomm) / Σ_red

    # Allocate
    nrealelem = length(grid.topology.realelems)
    δρ        = zeros(Nq * Nq * Nqk,1,nrealelem) 
    δρu       = zeros(Nq * Nq * Nqk,1,nrealelem)
    δρv       = zeros(Nq * Nq * Nqk,1,nrealelem)
    δρw       = zeros(Nq * Nq * Nqk,1,nrealelem) 
    δρe       = zeros(Nq * Nq * Nqk,1,nrealelem)
    δρq       = zeros(Nq * Nq * Nqk,1,nrealelem)
    tempmax   = zeros(nrealelem)
    (δρmax, δρumax, δρvmax, δρwmax, δρemax, δρqmax) = (zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),zero(FT))
    
    # Compute deviation from global means 
    for eh in 1:nhorzelem
      for ev in 1:nvertelem
        e = ev + (eh - 1) * nvertelem
        for k in 1:Nqk
          for j in 1:Nq
            for i in 1:Nq
              ijk          = i + Nq * ((j-1) + Nq * (k-1))
              δρ[ijk,1,e] = localQ[ijk,1,e] - ρ̅
              δρu[ijk,1,e] = localQ[ijk,2,e] - ρ̅u̅
              δρv[ijk,1,e] = localQ[ijk,3,e] - ρ̅v̅ 
              δρw[ijk,1,e] = localQ[ijk,4,e] - ρ̅w̅ 
              δρe[ijk,1,e] = localQ[ijk,5,e] - ρ̅e̅ 
              δρq[ijk,1,e] = localQ[ijk,6,e] - ρ̅q̅
            end
          end
        end
        δρmax  = maximum(δρ[:,1,e])
        δρumax = maximum(δρu[:,1,e])
        δρvmax = maximum(δρv[:,1,e])
        δρwmax = maximum(δρw[:,1,e])
        δρemax = maximum(δρe[:,1,e])
        δρqmax = maximum(δρq[:,1,e])
      end
    end
    @show(ρ̅)
    # Global maximum 
    return nothing
  end

  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                           )
  # -------------- Define model ---------------------------------- # 
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     DynamicSubgridStabilization(),
                     EquilMoist(), 
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

  Q = init_ode_state(dg, FT(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

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
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  cb_dynsgs = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
    #gather_global_stats(mpicomm, dg, Q, dt)
  end

  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(1)  do (init=false)
    mkpath("./vtk-rtb/")
      outprefix = @sprintf("./vtk-rtb/DC_%dD_mpirank%04d_step%04d", dim,
                           MPI.Comm_rank(mpicomm), step[1])
      @debug "doing VTK output" outprefix
      writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)), dg.auxstate, flattenednames(vars_aux(model,FT)))
      step[1] += 1
      nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo,cbvtk,cb_dynsgs))
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
  CLIMA.init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  for FT in (Float32, Float64)
    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
    engf_eng0 = run(mpicomm,
                    topl, dim, Ne, N, 
                    timeend, FT, dt)
    @test engf_eng0 ≈ FT(9.9999993807738441e-01)
  end
end
