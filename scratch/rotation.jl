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
                   HydrostaticState, NoOrientation
using CLIMA.VariableTemplates: flattenednames

using CLIMA.PlanetParameters: T_0
using CLIMA.DGmethods: VerticalDirection, DGModel, Vars, vars_state, num_state
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxDiffusive,
                                       CentralGradPenalty
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
  Nev = 1

  # Setup the topology
  brickrange = (range(FT(0); length=Neh+1, stop=1),
                range(FT(0); length=Neh+1, stop=1),
                range(FT(0); length=Nev+1, stop=1))
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity = (false, false, false))

  # Warp mesh and apply a random rotation
  (Q, _) = qr(rand(3,3))
  function warpfun(ξ1, ξ2, ξ3)
    ξ3 = 1 + ξ3 # + (ξ3 - 1) * sin(2π * ξ2 * ξ1) / 5
    @inbounds (ξ1, ξ2, ξ3)
  end
  function rotfun(ξ1, ξ2, ξ3)
    (ξ1, ξ2, ξ3) = warpfun(ξ1, ξ2, ξ3)

    ξ = SVector(ξ1, ξ2, ξ3)
    x = Q * ξ

    @inbounds (x[1], x[2], x[3])
  end

  # create the actual grid
  grid = (warp = DiscontinuousSpectralElementGrid(topl,
                                                  FloatType = FT,
                                                  DeviceArray = AT,
                                                  polynomialorder = N,
                                                  meshwarp = rotfun,
                                                 ),
          flat = DiscontinuousSpectralElementGrid(topl,
                                                  FloatType = FT,
                                                  DeviceArray = AT,
                                                  polynomialorder = N,
                                                  meshwarp = warpfun,
                                                 ))
  model = AtmosModel(NoOrientation(),
                     HydrostaticState(IsothermalProfile(FT(T_0)), FT(0)),
                     ConstantViscosityWithDivergence(0.0),
                     DryModel(),
                     NoRadiation(),
                     nothing,
                     NoFluxBC(),
                     nothing)
  linear_model = AtmosAcousticLinearModel(model)
  # the nonlinear model is needed so we can grab the auxstate below
  dg = (warp = DGModel(model,
                       grid.warp,
                       Rusanov(),
                       CentralNumericalFluxDiffusive(),
                       CentralGradPenalty()),
        flat = DGModel(model,
                       grid.flat,
                       Rusanov(),
                       CentralNumericalFluxDiffusive(),
                       CentralGradPenalty()))
  dg_linear = (warp = DGModel(linear_model,
                              grid.warp,
                              Rusanov(),
                              CentralNumericalFluxDiffusive(),
                              CentralGradPenalty();
                              direction=VerticalDirection(),
                              auxstate=dg.warp.auxstate),
               flat = DGModel(linear_model,
                              grid.flat,
                              Rusanov(),
                              CentralNumericalFluxDiffusive(),
                              CentralGradPenalty();
                              direction=VerticalDirection(),
                              auxstate=dg.flat.auxstate))

  A = (warp = I - SparseMatrixCSC(dg_linear.warp), 
       flat = I - SparseMatrixCSC(dg_linear.flat))
  println()

  nstate = num_state(linear_model, FT)
  Ndof = Nq^3 * nstate * Nev * Neh^2
  NdofV = Nq * Nev
  K = PermutedDimsArray(reshape(1:Ndof,  Nq, Nq, Nq, nstate, Nev, Neh^2),
                        (4, 3, 5, 1, 2, 6))
  ξ1x1, ξ2x1, ξ3x1 = grid.warp.vgeo[1, Grids._ξ1x1, 1], grid.warp.vgeo[1, Grids._ξ2x1, 1], grid.warp.vgeo[1, Grids._ξ3x1, 1]
  ξ1x2, ξ2x2, ξ3x2 = grid.warp.vgeo[1, Grids._ξ1x2, 1], grid.warp.vgeo[1, Grids._ξ2x2, 1], grid.warp.vgeo[1, Grids._ξ3x2, 1]
  ξ1x3, ξ2x3, ξ3x3 = grid.warp.vgeo[1, Grids._ξ1x3, 1], grid.warp.vgeo[1, Grids._ξ2x3, 1], grid.warp.vgeo[1, Grids._ξ3x3, 1]

  R = [ξ1x1 ξ1x2 ξ1x3;
       ξ2x1 ξ2x2 ξ2x3;
       ξ3x1 ξ3x2 ξ3x3] / 2
  display(R' * R)
  println()

  @assert size(K, 1) == nstate
  @assert size(K, 2) == Nq
  @assert size(K, 3) == Nev

  idof = 1
  jdof = 1
  helm = 1
  Ks = @view K[:, :, :, idof, jdof, helm][:]
  cA = (warp = A.warp[Ks, Ks], flat = A.flat[Ks, Ks])

  x = rand(nstate * NdofV)
  b = cA.warp * x
  y = cA.warp \ b

  @show norm(x - y)
  println()

  c = copy(b)
  for ev = 1:Nev
    for k = 1:Nq
      offset = ((ev-1) * Nq + k-1) * nstate
      state = Vars{vars_state(linear_model, FT)}(view(c, offset .+ (1:nstate)))
      state.ρu = R * state.ρu
    end
  end
  println()
  z = cA.flat \ c

  for ev = 1:Nev
    for k = 1:Nq
      offset = ((ev-1) * Nq + k-1) * nstate
      state = Vars{vars_state(linear_model, FT)}(view(z, offset .+ (1:nstate)))
      state.ρu = R' * state.ρu
    end
  end
  println()
  @show norm(z - y)
  @show norm(z)
  println()
  println()
  println()
  display(reshape(z, nstate, NdofV))
  println()
  display(reshape(y, nstate, NdofV))
  println()
  display(reshape(z - y, nstate, NdofV))
  println()
  # display(spy(cA.flat, width=nstate*NdofV, height=nstate*NdofV))
  println()

  # vtk for debugging
  x1 = reshape(view(grid.warp.vgeo, :, grid.warp.x1id, :), Nq, Nq, Nq, Neh^2*Nev)
  x2 = reshape(view(grid.warp.vgeo, :, grid.warp.x2id, :), Nq, Nq, Nq, Neh^2*Nev)
  x3 = reshape(view(grid.warp.vgeo, :, grid.warp.x3id, :), Nq, Nq, Nq, Neh^2*Nev)
  writemesh("mesh_warp", x1, x2, x3)
end

nothing
