using MPI
using Logging
using Test
using LinearAlgebra
using CLIMA
using CLIMA.MPIStateArrays
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

if !@isdefined integration_testing
  if length(ARGS) > 0
    const integration_testing = parse(Bool, ARGS[1])
  else
    const integration_testing =
      parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  end
end

const output = parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_OUTPUT","false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{n, α, β}, aux::Vars,
                                  geom::LocalGeometry) where {n, α, β}
  # Direction of flow is n with magnitude α
  aux.u = α * n

  # diffusion of strength β in the n direction
  aux.D = β * n * n'
end

function initial_condition!(::Pseudo1D{n, α, β, μ, δ}, state, aux, x,
                            t) where {n, α, β, μ, δ}
  ξn = dot(n, x)
  # ξT = SVector(x) - ξn * n
  state.ρ = exp(-(ξn - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
end

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

  polynomialorder = 4

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    # for FT in (Float64, Float32)
    #   for dim = 2:3
    for FT in (Float64, )
      for dim = 2
        d = dim == 2 ? FT[1, 10, 0] : FT[1, 1, 10]
        n = SVector{3, FT}(d ./ norm(d))

        α = FT(1)
        β = FT(1 // 100)
        μ = FT(-1 // 2)
        δ = FT(1 // 10)
        Ne = 5

        brickrange = (ntuple(j->range(FT(-1); length=Ne+1, stop=1), dim-1)...,
                      range(FT(-1); length=Ne+1, stop=1))

        periodicity = ntuple(j->false, dim)
        topl = StackedBrickTopology(mpicomm, brickrange;
                                    periodicity = periodicity,
                                    boundary = (ntuple(j->(1,1), dim-1)...,
                                                (3,3)))

        grid = DiscontinuousSpectralElementGrid(topl,
                                                FloatType = FT,
                                                DeviceArray = ArrayType,
                                                polynomialorder = polynomialorder)
        model = AdvectionDiffusion{dim}(Pseudo1D{n, α, β, μ, δ}())

        dge = DGModel(model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty(),
                      direction=DGmethods.EveryDirection())

        dgh = DGModel(model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty(),
                      auxstate=dge.auxstate,
                      direction=DGmethods.HorizontalDirection())

        dgv = DGModel(model,
                      grid,
                      Rusanov(),
                      CentralNumericalFluxDiffusive(),
                      CentralGradPenalty(),
                      auxstate=dge.auxstate,
                      direction=DGmethods.VerticalDirection())

        # Q = DGmethods.create_state(dge.balancelaw, dge.grid, 888)
        # Q.data .= rand(size(Q.data))

        Q = init_ode_state(dge, sqrt(FT(2)))

        dQe = similar(Q)
        dQh = similar(Q)
        dQv = similar(Q)

        t = sqrt(FT(2))

        dQe .= 0
        dQh .= 0
        dQv .= 0

        dge(dQe, Q, nothing, t; increment=false)
        dgh(dQh, Q, nothing, t; increment=false)
        dgv(dQv, Q, nothing, t; increment=false)

        dQhv =  dQh .+ dQv

        @test all(dQe.realdata .≈ dQhv.realdata)

        Q.data .= dQhv.data

        dge(dQe, Q, nothing, t; increment=false)
        dgh(dQh, Q, nothing, t; increment=false)
        dgv(dQv, Q, nothing, t; increment=false)

        dQhv =  dQh .+ dQv

        @show dQe.realdata .≈ dQhv.realdata

        @test all(dQe.realdata .≈ dQhv.realdata)
      end
    end
  end
end

nothing

