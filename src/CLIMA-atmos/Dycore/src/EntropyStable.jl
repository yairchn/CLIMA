module EntropyStable
include("vtk.jl")

using Requires
@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
  using .CUDAnative.CUDAdrv

  include("EntropyStable_cuda.jl")
end

using ..CLIMAAtmosDycore
AD = CLIMAAtmosDycore

using Canary
using MPI
using ParametersType
using PlanetParameters: cp_d, cv_d, R_d, grav
@parameter gamma_d cp_d/cv_d "Heat capcity ratio of dry air"
@parameter gdm1 R_d/cv_d "(equivalent to gamma_d-1)"

using Base: @kwdef

# {{{ constants

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                 x = _x,   y = _y,   z = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, nz = _nz, sMJ = _sMJ, vMJI = _vMJI)
# }}}

"""
    Parameters

Data structure containing the parameters for the vanilla DG discretization of
the compressible Euler equations

To get information about parameters do for example
`?Parameters.DFloat`

!!! note
Would be nice for the the docs to be recursively generated.
See Julia github issue [#25167](https://github.com/JuliaLang/julia/issues/25167)

"""
# {{{ Parameters
@kwdef struct Parameters # <: AD.AbstractSpaceParameter
  """
  Compute data type

  default: `Float64`
  """
  DFloat::Type = Float64

  """
  Device array type

  if 'Array' the cpu implemtation will be used

  if 'CuArray' the cuda implemtation will be used

  default: `Array`
  """
  DeviceArray::Type = Array

  """
  Function with arguments `(part, numparts)`, which returns initial partition
  `part` of the mesh of with number of partitions `numparts`

  no default value
  """
  meshgenerator::Function

  """
  Function to warp the coordinate points after mesh generation

  Syntax TBD

  default: `(x...)->identity(x)`
  """
  meshwarp::Function = (x...)->identity(x)

  """
  Intial time of the simulation

  default: 0
  """
  initialtime::AbstractFloat = 0

  """
  Number of spatial dimensiona

  default: `3`
  """
  dim::Int = 3

  """
  Boolean specifying whether or not to use gravity

  default: `true`
  """
  gravity::Bool = true

  """
  Polynomial order for discontinuous Galerkin method

  no default value
  """
  N::Int

  """
  number of moist variables

  default: 0
  """
  nmoist::Int = 0

  """
  number of tracer variables

  default: 0
  """
  ntrace::Int = 0
end
# }}}

"""
    Configuration

Data structure containing the configuration data for the vanilla DG
discretization of the compressible Euler equations
"""
# {{{ Configuration
struct Configuration{DeviceArray, HostArray} #<: AD.AbstractSpaceConfiguration
  "mpi communicator use for spatial discretization are using"
  mpicomm
  "mesh data structure from Canary"
  mesh
  "volume metric terms"
  vgeo::DeviceArray
  "surface metric terms"
  sgeo::DeviceArray
  "gravitational acceleration (m/s^2)"
  gravity
  "element to boundary condition map"
  elemtobndy::DeviceArray
  "volume DOF to element minus side map"
  vmapM::DeviceArray
  "volume DOF to element plus side map"
  vmapP::DeviceArray
  "list of elements that need to be communicated (in neighbors order)"
  sendelems::DeviceArray
  "MPI send request storage"
  sendreq::HostArray
  "MPI recv request storage"
  recvreq::HostArray
  "host storage for state to be sent"
  host_sendQ::HostArray
  "host storage for state to be recv'd"
  host_recvQ::HostArray
  "device storage for state to be sent"
  device_sendQ::DeviceArray
  "device storage for state to be recv'd"
  device_recvQ::DeviceArray
  "1-D derivative operator on the device"
  D::DeviceArray

  function Configuration(params::Parameters, mpicomm)
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    N = params.N
    dim = params.dim
    DFloat = params.DFloat

    mesh = params.meshgenerator(mpirank+1, mpisize)

    mpirank == 0 && @debug "partiting mesh..."
    mesh = partition(mpicomm, mesh...)

    # Connect the mesh in parallel
    mpirank == 0 && @debug "connecting mesh..."
    mesh = connectmesh(mpicomm, mesh...)

    # Get the vmaps
    mpirank == 0 && @debug "computing mappings..."
    (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                              mesh.elemtoordr)

    # Create 1-D operators
    (ξ, ω) = lglpoints(DFloat, N)
    D = spectralderivative(ξ)

    # Compute the geometry
    mpirank == 0 && @debug "computing metrics..."
    (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω,
                                   params.meshwarp, vmapM)
    gravity::DFloat = (params.gravity) ? grav : 0
    (nface, nelem) = size(mesh.elemtoelem)

    mpirank == 0 && @debug "create RHS storage..."

    mpirank == 0 && @debug "create send/recv request storage..."
    nnabr = length(mesh.nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)

    mpirank == 0 && @debug "create send/recv storage..."
    nvar = _nstate + params.nmoist + params.ntrace
    sendQ = zeros(DFloat, (N+1)^dim, nvar, length(mesh.sendelems))
    recvQ = zeros(DFloat, (N+1)^dim, nvar, length(mesh.ghostelems))

    mpirank == 0 && @debug "create configuration struct..."
    HostArray = Array
    DeviceArray = params.DeviceArray
    # FIXME: Handle better for GPU?
    new{DeviceArray, HostArray}(mpicomm, mesh, DeviceArray(vgeo),
                                DeviceArray(sgeo), gravity,
                                DeviceArray(mesh.elemtobndy),
                                DeviceArray(vmapM), DeviceArray(vmapP),
                                DeviceArray(mesh.sendelems), sendreq, recvreq,
                                sendQ, recvQ, DeviceArray(sendQ),
                                DeviceArray(recvQ), DeviceArray(D))
  end
end
# }}}

"""
    State

Data structure containing the state data for the vanilla DG discretization of
the compressible Euler equations
"""
# {{{ State
struct State{DeviceArray} #<: AD.AbstractSpaceState
  time
  Q::DeviceArray
  function State(config::Configuration{DeviceArray, HostArray},
                 params::Parameters) where {DeviceArray, HostArray}
    nvar = _nstate + params.nmoist + params.ntrace
    Q = similar(config.vgeo, (size(config.vgeo,1), nvar, size(config.vgeo,3)))
    # Shove into array so we can leave the type immutable
    # (is this worthwhile?)
    time = [params.initialtime]
    new{DeviceArray}(time, Q)
  end
end
# }}}

"""
    Runner

Data structure containing the runner for the vanilla DG discretization of
the compressible Euler equations

"""
# {{{ Runner
struct Runner{DeviceArray<:AbstractArray} <: AD.AbstractSpaceRunner
  params::Parameters
  config::Configuration
  state::State
  function Runner(mpicomm; args...)
    params = Parameters(;args...)
    config = Configuration(params, mpicomm)
    state = State(config, params)
    new{params.DeviceArray}(params, config, state)
  end
end
AD.createrunner(::Val{:EntropyStable}, m; a...) = Runner(m; a...)
# }}}

# {{{ show
function Base.show(io::IO, runner::Runner)
  eng = AD.L2solutionnorm(runner; host=true)
  print(io, "EntropyStable with norm2(Q) = ", eng, " at time = ",
        runner[:time])
end

function Base.show(io::IO, ::MIME"text/plain",
                   runner::Runner{DeviceArray}) where DeviceArray
  state = runner.state
  params = runner.params
  config = runner.config
  println(io, "EntropyStable with:")
  DFloat = eltype(state.Q)
  eng = AD.L2solutionnorm(runner; host=true)
  println(io, "   DeviceArray = ", DeviceArray)
  println(io, "   DFloat      = ", DFloat)
  println(io, "   norm2(Q)    = ", eng)
  println(io, "   time        = ", runner[:time])
  println(io, "   N           = ", params.N)
  println(io, "   dim         = ", params.dim)
  println(io, "   mpisize     = ", MPI.Comm_size(config.mpicomm))
end
# }}}

Base.getindex(r::Runner, s) = r[Symbol(s)]
function Base.getindex(r::Runner, s::Symbol)
  s == :time && return r.state.time[1]
  s == :Q && return r.state.Q
  s == :mesh && return r.config.mesh
  s == :stateid && return stateid
  s == :moistid && return _nstate .+ (1:r.params.nmoist)
  s == :traceid && return _nstate .+ r.params.nmoist .+ (1:r.params.ntrace)
  error("""
        getindex for the $(typeof(r)) supports:
        `:time`    => gets the runners time
        `:Q`       => gets the runners state Q
        `:mesh`    => gets the runners mesh
        `:hostQ`   => not implemented yet
        `:stateid` => Euler state storage order
        `:moistid` => moist state storage order
        `:traceid` => trace state storage order
        """)
end
Base.setindex!(r::Runner, v, s) = Base.setindex!(r, v, Symbol(s))
function Base.setindex!(r::Runner, v,  s::Symbol)
  s == :time && return r.state.time[1] = v
  error("""
        setindex! for the $(typeof(r)) supports:
        `:time` => sets the runners time
        """)
end

# {{{ initstate!
function AD.initstate!(runner::Runner{DeviceArray}, ic::Function;
                       host=false) where DeviceArray

  host || error("Currently requires host configuration")

  # Pull out the config and state
  params::Parameters = runner.params
  config::Configuration = runner.config
  state::State = runner.state

  # Get the number of elements
  cpubackend = DeviceArray == Array
  vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  Q = cpubackend ? state.Q : Array(state.Q)

  nvar = _nstate + params.nmoist + params.ntrace
  @inbounds for e = 1:size(Q, 3), i = 1:size(Q, 1)
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    Q0 = ic(x, y, z)
    for n = 1:nvar
      Q[i, n, e] = Q0[n]
    end
  end
  if !cpubackend
    state.Q .= Q
  end
end
#}}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(mesh.elemtoelem)

  # crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

  vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  creategrid!(X..., mesh.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
  end

  # Compute the metric terms
  if dim == 1
    computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  elseif dim == 3
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
  sMJ .= sM .* sJ

  (vgeo, sgeo)
end
# }}}

# {{{ cfl
function AD.estimatedt(runner::Runner{DeviceArray};
                       host=false) where {DeviceArray}
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  Q = cpubackend ? state.Q : Array(state.Q)
  estimatedt(Val(params.dim), Val(params.N), vgeo, config.gravity, Q,
             config.mpicomm)
end

function estimatedt(::Val{dim}, ::Val{N}, vgeo, gravity, Q,
                    mpicomm) where {dim, N}
  DFloat = eltype(Q)

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
      E = Q[n, _E, e]
      y = vgeo[n, _y, e]
      P = gdm1*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * sqrt(gamma_d * P / ρ),
                        abs(U * ηx + V * ηy) + ρ * sqrt(gamma_d * P / ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      E = Q[n, _E, e]
      z = vgeo[n, _z, e]
      P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(gamma_d*P/ρ),
                        abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(gamma_d*P/ρ))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm) / N^√2
end
#}}}

# {{{ writevtk
function AD.writevtk(runner::Runner{DeviceArray}, prefix;
                     Q = nothing, vgeo = nothing) where DeviceArray
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  if vgeo == nothing
    vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  end
  if Q == nothing
    Q = cpubackend ? state.Q : Array(state.Q)
  end

  Nq  = params.N+1
  dim = params.dim

  nelem = size(Q)[end]
  X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->Nq,dim)...,
                        nelem), dim)
  ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->Nq, dim)..., nelem)
  U = reshape((@view Q[:, _U, :]), ntuple(j->Nq, dim)..., nelem)
  V = reshape((@view Q[:, _V, :]), ntuple(j->Nq, dim)..., nelem)
  W = reshape((@view Q[:, _W, :]), ntuple(j->Nq, dim)..., nelem)
  E = reshape((@view Q[:, _E, :]), ntuple(j->Nq, dim)..., nelem)
  writemesh(prefix, X...;
            fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
            realelems=config.mesh.realelems)
end
# }}}

# {{{ L2 Energy (for all dimensions)
function AD.L2solutionnorm(runner::Runner{DeviceArray};
                           host=false, Q = nothing, vgeo = nothing
                          ) where DeviceArray
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  if vgeo == nothing
    vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  end
  if Q == nothing
    Q = cpubackend ? state.Q : Array(state.Q)
  end

  dim = params.dim
  N = params.N
  realelems = config.mesh.realelems
  locnorm2 = L2solutionnorm(Val(dim), Val(N), Q, vgeo, realelems)
  sqrt(MPI.allreduce([locnorm2], MPI.SUM, config.mpicomm)[1])
end

function L2solutionnorm(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
  end

  energy
end
# }}}

# {{{ L2 Error (for all dimensions)
function AD.L2errornorm(runner::Runner{DeviceArray}, Qexact;
                        host=false, Q = nothing, vgeo = nothing,
                        time = nothing) where DeviceArray
  host || error("Currently requires host configuration")
  state = runner.state
  config = runner.config
  params = runner.params
  cpubackend = DeviceArray == Array
  if vgeo == nothing
    vgeo = cpubackend ? config.vgeo : Array(config.vgeo)
  end
  if Q == nothing
    Q = cpubackend ? state.Q : Array(state.Q)
  end
  if time == nothing
    time = state.time[1]
  end

  dim = params.dim
  N = params.N
  realelems = config.mesh.realelems
  locnorm2 = L2errornorm(Val(dim), Val(N), time, Q, vgeo, realelems, Qexact)
  sqrt(MPI.allreduce([locnorm2], MPI.SUM, config.mpicomm)[1])
end

function L2errornorm(::Val{dim}, ::Val{N}, time, Q, vgeo, elems,
                     Qexact) where
  {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  errorsq = zero(DFloat)

  @inbounds for e = elems,  i = 1:Np
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    ρex, Uex, Vex, Wex, Eex = Qexact(time, x, y, z)

    errorsq += vgeo[i, _MJ, e] * (Q[i, _ρ, e] - ρex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _U, e] - Uex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _V, e] - Vex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _W, e] - Wex)^2
    errorsq += vgeo[i, _MJ, e] * (Q[i, _E, e] - Eex)^2
  end

  errorsq
end
# }}}

# {{{ RHS function
function AD.rhs!(rhs::DeviceArray,
                 runner::Runner{DeviceArray}) where DeviceArray
  state = runner.state
  config = runner.config
  params = runner.params
  mesh = config.mesh
  mpicomm = config.mpicomm
  sendreq = config.sendreq
  recvreq = config.recvreq
  host_recvQ = config.host_recvQ
  host_sendQ = config.host_sendQ
  device_recvQ = config.device_recvQ
  device_sendQ = config.device_sendQ
  sendelems = config.sendelems

  vgeo = config.vgeo
  sgeo = config.sgeo
  Dmat = config.D
  vmapM = config.vmapM
  vmapP = config.vmapP
  gravity = config.gravity
  elemtobndy = config.elemtobndy

  Q = state.Q

  N   = params.N
  dim = params.dim
  ntrace = params.ntrace
  nmoist = params.nmoist

  nnabr = length(mesh.nabrtorank)
  nrealelem = length(mesh.realelems)

  # post MPI receives
  for n = 1:nnabr
    recvreq[n] = MPI.Irecv!((@view host_recvQ[:, :, mesh.nabrtorecv[n]]),
                            mesh.nabrtorank[n], 777, mpicomm)
  end

  # wait on (prior) MPI sends
  MPI.Waitall!(sendreq)

  # pack data in send buffer
  fillsendQ!(host_sendQ, device_sendQ, Q, sendelems)

  # post MPI sends
  for n = 1:nnabr
    sendreq[n] = MPI.Isend((@view host_sendQ[:, :, mesh.nabrtosend[n]]),
                           mesh.nabrtorank[n], 777, mpicomm)
  end

  # volume RHS computation
  volumerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), rhs, Q, vgeo, gravity,
             Dmat, mesh.realelems)

  # wait on MPI receives
  MPI.Waitall!(recvreq)

  # copy data to state vectors
  transferrecvQ!(device_recvQ, host_recvQ, Q, nrealelem)

  # face RHS computation
  facerhs!(Val(dim), Val(N), Val(nmoist), Val(ntrace), rhs, Q, vgeo, sgeo,
           gravity, mesh.realelems, vmapM, vmapP, elemtobndy)
end
# }}}

# {{{ MPI Buffer handling
function fillsendQ!(host_sendQ, device_sendQ::Array, Q, sendelems)
  host_sendQ[:, :, :] .= Q[:, :, sendelems]
end

function transferrecvQ!(device_recvQ::Array, host_recvQ, Q, nrealelem)
  Q[:, :, nrealelem+1:end] .= host_recvQ[:, :, :]
end
# }}}

function aln(L, R)
  ζ = L / R
  f = (ζ - 1) / (ζ + 1)
  u = f * f
  ϵ = eltype(L)(1e-2)
  F = (u < ϵ) ?  F = 1 + u / 3 + u^2 / 5 + u^3 / 7 : log(ζ) / (2f)
  (L + R) / 2F
end

function flux!(F, UM, VM, WM, ρM, EM, zM, UP, VP, WP, ρP, EP, zP, gravity)
  ρMinv, ρPinv = 1 / ρM, 1/ρP
  uM, vM, wM = UM * ρMinv, VM * ρMinv, WM * ρMinv
  uP, vP, wP = UP * ρPinv, VP * ρPinv, WP * ρPinv

  PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*zM)
  PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*zP)
  βM = ρM / 2PM
  βP = ρP / 2PP

  ua  = (uM + uP) / 2
  va  = (vM + vP) / 2
  wa  = (wM + wP) / 2
  u2a = ((uM^2 + vM^2 + wM^2) + (uP^2 + vP^2 + wP^2)) / 2
  ρa  = (ρM + ρP) / 2
  βa = (βM + βP) / 2
  ρln = aln(ρM, ρP)
  βln = aln(βM, βP)
  ϕa = gravity * (zM + zP) / 2

  F[_ρ, 1] = ρln * ua
  F[_ρ, 2] = ρln * va
  F[_ρ, 3] = ρln * wa

  F[_U, 1] = F[_ρ, 1] * ua + ρa / 2βa
  F[_V, 1] = F[_ρ, 1] * va
  F[_W, 1] = F[_ρ, 1] * wa

  F[_U, 2] = F[_ρ, 2] * ua
  F[_V, 2] = F[_ρ, 2] * va + ρa / 2βa
  F[_W, 2] = F[_ρ, 2] * wa

  F[_U, 3] = F[_ρ, 3] * ua
  F[_V, 3] = F[_ρ, 3] * va
  F[_W, 3] = F[_ρ, 3] * wa + ρa / 2βa

  Efac = 1 / (2*gdm1*βln) - u2a / 2 + ϕa
  F[_E, 1] = (F[_U, 1] * ua + F[_V, 1] * va + F[_W, 1] * wa) + Efac * F[_ρ, 1]
  F[_E, 2] = (F[_U, 2] * ua + F[_V, 2] * va + F[_W, 2] * wa) + Efac * F[_ρ, 2]
  F[_E, 3] = (F[_U, 3] * ua + F[_V, 3] * va + F[_W, 3] * wa) + Efac * F[_ρ, 3]
end


# {{{ Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N, nmoist, ntrace}
  nvar = _nstate + nmoist + ntrace

  # Moist and tracers not implemented yet
  @assert nmoist == 0
  @assert ntrace == 0

  DFloat = eltype(Q)

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, nvar, nelem)
  rhs = reshape(rhs, Nq, Nq, nvar, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  F       = Array{DFloat}(undef, _nstate, 3)
  l_MJrhs = Array{DFloat}(undef, _nstate)

  @inbounds for e in elems
    for j = 1:Nq, i = 1:Nq
      Uij, Vij = Q[i, j, _U, e], Q[i, j, _V, e]
      ρij, Eij = Q[i, j, _ρ, e], Q[i, j, _E, e]
      Wij = zero(DFloat)

      MJij = vgeo[i, j, _MJ, e]
      ξxij, ξyij = vgeo[i, j, _ξx, e], vgeo[i, j, _ξy, e]
      ηxij, ηyij = vgeo[i, j, _ηx, e], vgeo[i, j, _ηy, e]
      yorzij = vgeo[i, j, _y, e]

      l_MJrhs  .= 0
      for n = 1:Nq
        Unj, Vnj = Q[n, j, _U, e], Q[n, j, _V, e]
        ρnj, Enj = Q[n, j, _ρ, e], Q[n, j, _E, e]
        Wnj = zero(DFloat)
        yorznj = vgeo[n, j, _y, e]

        MJnj = vgeo[n, j, _MJ, e]
        ξxnj, ξynj = vgeo[n, j, _ξx, e], vgeo[n, j, _ξy, e]
        yorznj = vgeo[n, j, _y, e]

        flux!(F, Uij, Vij, Wij, ρij, Eij, yorzij,
                 Unj, Vnj, Wnj, ρnj, Enj, yorznj,
                 gravity)
        for s = 1:_nstate
          # J W ξ_{x} (D_ξ ∘ F_{sx}) 1⃗ + J W ξ_{y} (D_ξ ∘ F_{sy}) 1⃗
          l_MJrhs[s] += MJij * ξxij * D[i, n] * F[s, 1]
          l_MJrhs[s] += MJij * ξyij * D[i, n] * F[s, 2]
          # (F_{sx} ∘ D_ξ^T) J W ξ_{x} 1⃗ + (F_{sy} ∘ D_ξ^T) J W ξ_{y} 1⃗
          l_MJrhs[s] -= D[n, i] * F[s, 1] * MJnj * ξxnj
          l_MJrhs[s] -= D[n, i] * F[s, 2] * MJnj * ξynj
        end
      end

      for n = 1:Nq
        Uin, Vin = Q[i, n, _U, e], Q[i, n, _V, e]
        ρin, Ein = Q[i, n, _ρ, e], Q[i, n, _E, e]
        Win = zero(DFloat)
        yorzin = vgeo[i, n, _y, e]

        MJin = vgeo[i, n, _MJ, e]
        ηxin, ηyin = vgeo[i, n, _ηx, e], vgeo[i, n, _ηy, e]
        yorzin = vgeo[i, n, _y, e]

        flux!(F, Uij, Vij, Wij, ρij, Eij, yorzij,
                 Uin, Vin, Win, ρin, Ein, yorzin,
                 gravity)
        for s = 1:_nstate
          # J W η_{x} (D_η ∘ F_{sx}) 1⃗ + J W η_{y} (D_η ∘ F_{sy}) 1⃗
          l_MJrhs[s] += MJij * ηxij * D[j, n] * F[s, 1]
          l_MJrhs[s] += MJij * ηyij * D[j, n] * F[s, 2]
          # (F_{sx} ∘ D_η^T) J W η_{x} 1⃗ + (F_{sy} ∘ D_η^T) J W η_{y} 1⃗
          l_MJrhs[s] -= D[n, j] * F[s, 1] * MJin * ηxin
          l_MJrhs[s] -= D[n, j] * F[s, 2] * MJin * ηyin
        end
      end

      MJI = vgeo[i, j, _MJI, e]
      for s = 1:_nstate
        rhs[i, j, s, e] -= MJI * l_MJrhs[s]
      end

      # FIXME: buoyancy term
      rhs[i, j, _V, e] -= ρij * gravity
    end
  end
end
# }}}

# {{{ Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                    rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N, nmoist, ntrace}
  nvar = _nstate + nmoist + ntrace

  # Moist and tracers not implemented yet
  @assert nmoist == 0
  @assert ntrace == 0

  DFloat = eltype(Q)

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, Nq, nvar, nelem)
  rhs = reshape(rhs, Nq, Nq, Nq, nvar, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

  F       = Array{DFloat}(undef, _nstate, 3)
  l_MJrhs = Array{DFloat}(undef, _nstate)

  @inbounds for e in elems
    for k = 1:Nq, j = 1:Nq, i = 1:Nq

      #{{{ loads for ijk
      ρijk = Q[i, j, k, _ρ, e]
      Uijk = Q[i, j, k, _U, e]
      Vijk = Q[i, j, k, _V, e]
      Wijk = Q[i, j, k, _W, e]
      Eijk = Q[i, j, k, _E, e]

      MJijk = vgeo[i, j, k, _MJ, e]
      ξxijk = vgeo[i, j, k, _ξx, e]
      ξyijk = vgeo[i, j, k, _ξy, e]
      ξzijk = vgeo[i, j, k, _ξz, e]
      ηxijk = vgeo[i, j, k, _ηx, e]
      ηyijk = vgeo[i, j, k, _ηy, e]
      ηzijk = vgeo[i, j, k, _ηz, e]
      ζxijk = vgeo[i, j, k, _ζx, e]
      ζyijk = vgeo[i, j, k, _ζy, e]
      ζzijk = vgeo[i, j, k, _ζz, e]
      yorzijk = vgeo[i, j, k, _z, e]
      #}}}

      l_MJrhs  .= 0
      for n = 1:Nq
        # ξ-direction
        #{{{ loads for njk
        #
        ρnjk = Q[n, j, k, _ρ, e]
        Unjk = Q[n, j, k, _U, e]
        Vnjk = Q[n, j, k, _V, e]
        Wnjk = Q[n, j, k, _W, e]
        Enjk = Q[n, j, k, _E, e]

        MJnjk = vgeo[n, j, k, _MJ, e]
        ξxnjk = vgeo[n, j, k, _ξx, e]
        ξynjk = vgeo[n, j, k, _ξy, e]
        ξznjk = vgeo[n, j, k, _ξz, e]
        yorznjk = vgeo[n, j, k, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Unjk, Vnjk, Wnjk, ρnjk, Enjk, yorznjk,
                 gravity)
        for s = 1:_nstate
          # J W ξ_{x} (D_ξ ∘ F_{sx}) 1⃗ + J W ξ_{y} (D_ξ ∘ F_{sy}) 1⃗ +
          # J W ξ_{z} (D_ξ ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ξxijk * D[i, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ξyijk * D[i, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ξzijk * D[i, n] * F[s, 3]
          # (F_{sx} ∘ D_ξ^T) J W ξ_{x} 1⃗ + (F_{sy} ∘ D_ξ^T) J W ξ_{y} 1⃗ +
          # (F_{sz} ∘ D_ξ^T) J W ξ_{z} 1⃗
          l_MJrhs[s] -= D[n, i] * F[s, 1] * MJnjk * ξxnjk
          l_MJrhs[s] -= D[n, i] * F[s, 2] * MJnjk * ξynjk
          l_MJrhs[s] -= D[n, i] * F[s, 3] * MJnjk * ξznjk
        end

        # η-direction
        #{{{ loads for ink
        ρink = Q[i, n, k, _ρ, e]
        Uink = Q[i, n, k, _U, e]
        Vink = Q[i, n, k, _V, e]
        Wink = Q[i, n, k, _W, e]
        Eink = Q[i, n, k, _E, e]

        MJink = vgeo[i, n, k, _MJ, e]
        ηxink = vgeo[i, n, k, _ηx, e]
        ηyink = vgeo[i, n, k, _ηy, e]
        ηzink = vgeo[i, n, k, _ηz, e]
        yorzink = vgeo[i, n, k, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Uink, Vink, Wink, ρink, Eink, yorzink,
                 gravity)
        for s = 1:_nstate
          # J W η_{x} (D_η ∘ F_{sx}) 1⃗ + J W η_{y} (D_η ∘ F_{sy}) 1⃗ +
          # J W η_{z} (D_η ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ηxijk * D[j, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ηyijk * D[j, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ηzijk * D[j, n] * F[s, 3]
          # (F_{sx} ∘ D_η^T) J W η_{x} 1⃗ + (F_{sy} ∘ D_η^T) J W η_{y} 1⃗ +
          # (F_{sz} ∘ D_η^T) J W η_{z} 1⃗
          l_MJrhs[s] -= D[n, j] * F[s, 1] * MJink * ηxink
          l_MJrhs[s] -= D[n, j] * F[s, 2] * MJink * ηyink
          l_MJrhs[s] -= D[n, j] * F[s, 3] * MJink * ηzink
        end

        # ζ-direction
        #{{{ loads for ijn
        ρijn = Q[i, j, n, _ρ, e]
        Uijn = Q[i, j, n, _U, e]
        Vijn = Q[i, j, n, _V, e]
        Wijn = Q[i, j, n, _W, e]
        Eijn = Q[i, j, n, _E, e]

        MJijn = vgeo[i, j, n, _MJ, e]
        ζxijn = vgeo[i, j, n, _ζx, e]
        ζyijn = vgeo[i, j, n, _ζy, e]
        ζzijn = vgeo[i, j, n, _ζz, e]
        yorzijn = vgeo[i, j, n, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Uijn, Vijn, Wijn, ρijn, Eijn, yorzijn,
                 gravity)
        for s = 1:_nstate
          # J W ζ_{x} (D_ζ ∘ F_{sx}) 1⃗ + J W ζ_{y} (D_ζ ∘ F_{sy}) 1⃗ +
          # J W ζ_{z} (D_ζ ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ζxijk * D[k, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ζyijk * D[k, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ζzijk * D[k, n] * F[s, 3]
          # (F_{sx} ∘ D_ζ^T) J W ζ_{x} 1⃗ + (F_{sy} ∘ D_ζ^T) J W ζ_{y} 1⃗ +
          # (F_{sz} ∘ D_ζ^T) J W ζ_{z} 1⃗
          l_MJrhs[s] -= D[n, k] * F[s, 1] * MJijn * ζxijn
          l_MJrhs[s] -= D[n, k] * F[s, 2] * MJijn * ζyijn
          l_MJrhs[s] -= D[n, k] * F[s, 3] * MJijn * ζzijn
        end
      end

      MJI = vgeo[i, j, k, _MJI, e]
      for s = 1:_nstate
        rhs[i, j, k, s, e] -= MJI * l_MJrhs[s]
      end

      # FIXME: buoyancy term
      rhs[i, j, _W, e] -= ρijk * gravity
    end
  end
end
# }}}

# {{{ Face RHS (all dimensions)
function facerhs!(::Val{dim}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                  rhs::Array, Q, vgeo, sgeo, gravity, elems, vmapM, vmapP,
                  elemtobndy) where {N, dim, nmoist, ntrace}
  DFloat = eltype(Q)

  # Moist and tracers not implemented yet
  @assert nmoist == 0
  @assert ntrace == 0

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  F = similar(Q, (_nstate, 3))

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        (nxM, nyM, nzM, sMJ, vMJI) = sgeo[:, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        WM = Q[vidM, _W, eM]
        EM = Q[vidM, _E, eM]
        yorzM = (dim == 2) ? vgeo[vidM, _y, eM] : vgeo[vidM, _z, eM]

        bc = elemtobndy[f, e]
        PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*yorzM)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          WP = Q[vidP, _W, eP]
          EP = Q[vidP, _E, eP]
          yorzP = (dim == 2) ? vgeo[vidP, _y, eP] : vgeo[vidP, _z, eP]
          PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*yorzP)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM + nzM * WM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          WP = WM - 2 * UnM * nzM
          ρP = ρM
          EP = EM
          yorzP = yorzM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        flux!(F, UM, VM, WM, ρM, EM, yorzM, UP, VP, WP, ρP, EP, yorzP, gravity)

        #Compute Numerical Flux and Update
        fluxUS = nxM * F[_U, 1] + nyM * F[_U, 2] + nzM * F[_U, 3]
        fluxVS = nxM * F[_V, 1] + nyM * F[_V, 2] + nzM * F[_V, 3]
        fluxWS = nxM * F[_W, 1] + nyM * F[_W, 2] + nzM * F[_W, 3]
        fluxρS = nxM * F[_ρ, 1] + nyM * F[_ρ, 2] + nzM * F[_ρ, 3]
        fluxES = nxM * F[_E, 1] + nyM * F[_E, 2] + nzM * F[_E, 3]

        #Update RHS
        rhs[vidM, _U, eM] -= vMJI * sMJ * fluxUS
        rhs[vidM, _V, eM] -= vMJI * sMJ * fluxVS
        rhs[vidM, _W, eM] -= vMJI * sMJ * fluxWS
        rhs[vidM, _ρ, eM] -= vMJI * sMJ * fluxρS
        rhs[vidM, _E, eM] -= vMJI * sMJ * fluxES
      end
    end
  end
end
# }}}

end
