using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.VTK: writemesh
using Logging
using LinearAlgebra
using Random
Random.seed!(777)
using StaticArrays
using CLIMA.Atmos: AtmosModel, AtmosAcousticLinearModel,
                   DryModel, NoRadiation, NoFluxBC,
                   ConstantViscosityWithDivergence, IsothermalProfile,
                   HydrostaticState, FlatOrientation
using CLIMA.VariableTemplates: flattenednames

using CLIMA.PlanetParameters: T_0
using CLIMA.DGmethods: VerticalDirection, DGModel, Matrix
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxDiffusive,
                                       CentralGradPenalty, vars_aux
using SparseArrays
using UnicodePlots


let
  # boiler plate MPI stuff
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  # Floating point type
  FT = Float64

  # Array type
  AT = Array

  # Mesh generation parameters
  N = 4
  Nq = N+1
  Neh = 1
  Nev = 5

  # Setup the topology
  brickrange = (range(FT(0); length=Neh+1, stop=1),
                range(FT(0); length=Neh+1, stop=1),
                range(FT(0); length=Nev+1, stop=1))
  topl = BrickTopology(mpicomm, brickrange, periodicity = (false, false, false))

  # Warp mesh and apply a random rotation
  (Q, R) = qr(rand(3,3))
  function warpfun(ξ1, ξ2, ξ3)
    ξ3 = 1 + ξ3 + (ξ3 - 1) * sin(2π * ξ2 * ξ1) / 5

    ξ = SVector(ξ1, ξ2, ξ3)
    x = Q * ξ

    @inbounds (x[1], x[2], x[3])
  end

  # create the actual grid
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = AT,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  model = AtmosModel(FlatOrientation(),
                     HydrostaticState(IsothermalProfile(FT(T_0)), FT(0)),
                     ConstantViscosityWithDivergence(0.0),
                     DryModel(),
                     NoRadiation(),
                     nothing,
                     NoFluxBC(),
                     nothing)
  linear_model = AtmosAcousticLinearModel(model)
  @show flattenednames(vars_aux(linear_model, Float64))
  # the nonlinear model is needed so we can grab the auxstate below
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())
  dg_linear = DGModel(linear_model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty();
                      direction=VerticalDirection(),
                      auxstate=dg.auxstate)

  A = SparseMatrixCSC(dg_linear)
  display(spy(A))

  # vtk for debugging
  x1 = reshape(view(grid.vgeo, :, grid.x1id, :), Nq, Nq, Nq, Neh^2*Nev)
  x2 = reshape(view(grid.vgeo, :, grid.x2id, :), Nq, Nq, Nq, Neh^2*Nev)
  x3 = reshape(view(grid.vgeo, :, grid.x3id, :), Nq, Nq, Nq, Neh^2*Nev)
  writemesh("mesh", x1, x2, x3)

end

nothing
