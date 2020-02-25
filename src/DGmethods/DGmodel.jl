struct DGModel{BL,G,NFND,NFD,GNF,AS,DS,D,MD}
  balancelaw::BL
  grid::G
  numfluxnondiff::NFND
  numfluxdiff::NFD
  gradnumflux::GNF
  auxstate::AS
  diffstate::DS
  direction::D
  modeldata::MD
end
function DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 auxstate=create_auxstate(balancelaw, grid),
                 diffstate=create_diffstate(balancelaw, grid),
                 direction=EveryDirection(), modeldata=nothing)
  DGModel(balancelaw, grid, numfluxnondiff, numfluxdiff, gradnumflux, auxstate,
          diffstate, direction, modeldata)
end

function (dg::DGModel)(dQdt, Q, ::Nothing, t; increment=false)
  bl = dg.balancelaw
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Nfp = Nq * Nqk
  nrealelem = length(topology.realelems)

  Qvisc = dg.diffstate
  auxstate = dg.auxstate

  FT = eltype(Q)
  nviscstate = num_diffusive(bl, FT)

  Np = dofs_per_element(grid)

  communicate = !(isstacked(topology) &&
                  typeof(dg.direction) <: VerticalDirection)

  aux_comm = update_aux!(dg, bl, Q, t)
  @assert typeof(aux_comm) == Bool

  # The pattern to overlap communication and computation
  # [device: copy] Start device to host data transfer
  # [device: cmdx] Start the volume kernel (and maybe some face?)
  # [host        ] Start MPI Sends (wait on copy stream)
  # [host        ] Wait  MPI Recv
  # [device: copy] transfer data from device to host and fill Q -> event (a)
  # [device: cmdx] Wait on event (a)
  # [device: cmdx] do face computation

  if device == CUDA()
    copy_stream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
    cmdx_stream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
  else
    copy_stream = cmdx_stream = nothing
  end

  ########################
  # Gradient Computation #
  ########################
  if communicate
    MPIStateArrays.start_ghost_data_transfer!(Q, stream=copy_stream, async=true)
    aux_comm && MPIStateArrays.start_ghost_data_transfer!(auxstate,
                                                          stream=copy_stream,
                                                          async=true)
  end

  if nviscstate > 0
    if device == CPU() && communicate
      MPIStateArrays.start_ghost_send!(Q)
      aux_comm && MPIStateArrays.start_ghost_send!(auxstate)
    end

    @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem, stream=cmdx_stream,
            volumeviscterms!(bl, Val(dim), Val(N), dg.direction, Q.data,
                             Qvisc.data, auxstate.data, grid.vgeo, t, grid.D,
                             topology.realelems))

    if device == CUDA() && communicate
      friendlysynchronize(copy_stream)
      MPIStateArrays.start_ghost_send!(Q)
      aux_comm && MPIStateArrays.start_ghost_send!(auxstate)
    end
    if communicate
      MPIStateArrays.finish_ghost_recv!(Q; stream=copy_stream, async=true)
      aux_comm && MPIStateArrays.finish_ghost_recv!(auxstate;
                                                    stream=copy_stream,
                                                    async=true)

      # have the cmdx_stream wait on the copy_stream
      if device == CUDA()
        event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
        CUDAdrv.record(event, copy_stream)
        CUDAdrv.wait(event, cmdx_stream)
      end
    end

    @launch(device, threads=Nfp, blocks=nrealelem, stream=cmdx_stream,
            faceviscterms!(bl, Val(dim), Val(N), dg.direction,
                           dg.gradnumflux, Q.data, Qvisc.data, auxstate.data,
                           grid.vgeo, grid.sgeo, t, grid.vmapM, grid.vmapP,
                           grid.elemtobndy,
                           topology.realelems))

    communicate && MPIStateArrays.start_ghost_data_transfer!(Qvisc,
                                                             stream=copy_stream,
                                                             async=true)

    aux_comm = update_aux_diffusive!(dg, bl, Q, t)
    @assert typeof(aux_comm) == Bool

    aux_comm && MPIStateArrays.start_ghost_data_transfer!(auxstate,
                                                          stream=copy_stream,
                                                          async=true)
  end


  ###################
  # RHS Computation #
  ###################
  @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem, stream=cmdx_stream,
          volumerhs!(bl, Val(dim), Val(N), dg.direction, dQdt.data,
                     Q.data, Qvisc.data, auxstate.data, grid.vgeo, t,
                     grid.ω, grid.D, topology.realelems, increment))

  if communicate && device == CUDA()
    friendlysynchronize(copy_stream)
    MPIStateArrays.start_ghost_send!((nviscstate > 0) ? Qvisc : Q)
    aux_comm && MPIStateArrays.start_ghost_send!(auxstate)
  end

  @launch(device, threads=Nfp, blocks=nrealelem, stream=cmdx_stream,
          facerhs!(bl, Val(dim), Val(N), dg.direction,
                   dg.numfluxnondiff,
                   dg.numfluxdiff,
                   dQdt.data, Q.data, Qvisc.data,
                   auxstate.data, grid.vgeo, grid.sgeo, t, grid.vmapM,
                   grid.vmapP, grid.elemtobndy,
                   topology.realelems))

  if communicate && device == CPU()
    MPIStateArrays.start_ghost_send!((nviscstate > 0) ? Qvisc : Q)
    aux_comm && MPIStateArrays.start_ghost_send!(auxstate)
  end
  if communicate
    MPIStateArrays.finish_ghost_recv!((nviscstate > 0) ? Qvisc : Q,
                                      stream=copy_stream,
                                      async=true)
    aux_comm && MPIStateArrays.finish_ghost_recv!(auxstate,
                                                  stream=copy_stream,
                                                  async=true)

    # have the cmdx_stream wait on the copy_stream
    if device == CUDA()
      event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
      CUDAdrv.record(event, copy_stream)
      CUDAdrv.wait(event, cmdx_stream)
    end
  end

  # Just to be safe, we wait on the sends we started.
  if communicate
    MPIStateArrays.finish_ghost_send!(Qvisc)
    MPIStateArrays.finish_ghost_send!(Q)
    MPIStateArrays.finish_ghost_send!(auxstate)
  end
end

function init_ode_state(dg::DGModel, args...;
                        forcecpu=false,
                        commtag=888)
  device = arraytype(dg.grid) <: Array ? CPU() : CUDA()

  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  topology = grid.topology
  Np = dofs_per_element(grid)

  auxstate = dg.auxstate
  dim = dimensionality(grid)
  N = polynomialorder(grid)
  nrealelem = length(topology.realelems)

  if !forcecpu
    @launch(device, threads=(Np,), blocks=nrealelem,
            initstate!(bl, Val(dim), Val(N), state.data, auxstate.data, grid.vgeo,
                     topology.realelems, args...))
  else
    h_state = similar(state, Array)
    h_auxstate = similar(auxstate, Array)
    h_auxstate .= auxstate
    @launch(CPU(), threads=(Np,), blocks=nrealelem,
      initstate!(bl, Val(dim), Val(N), h_state.data, h_auxstate.data, Array(grid.vgeo),
          topology.realelems, args...))
    state .= h_state
  end

  MPIStateArrays.start_ghost_exchange!(state)
  MPIStateArrays.finish_ghost_exchange!(state)

  return state
end

# fallback
function update_aux!(dg::DGModel, bl::BalanceLaw, Q::MPIStateArray, t::Real)
  return false
end

function update_aux_diffusive!(dg::DGModel, bl::BalanceLaw, Q::MPIStateArray, t::Real)
  return false
end

function indefinite_stack_integral!(dg::DGModel, m::BalanceLaw,
                                    Q::MPIStateArray,
                                    auxstate::MPIStateArray,
                                    t::Real)

  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(Q)

  # do integrals
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_indefinite_stack_integral!(m, Val(dim), Val(N),
                                         Val(nvertelem),
                                         Q.data, auxstate.data,
                                         grid.vgeo, grid.Imat, 1:nhorzelem))
end

function reverse_indefinite_stack_integral!(dg::DGModel,
                                            m::BalanceLaw,
                                            Q::MPIStateArray,
                                            auxstate::MPIStateArray, t::Real)

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  FT = eltype(auxstate)

  # do integrals
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_reverse_indefinite_stack_integral!(m, Val(dim), Val(N),
                                                 Val(nvertelem),
                                                 Q.data, auxstate.data,
                                                 1:nhorzelem))
end

function nodal_update_aux!(f!, dg::DGModel, m::BalanceLaw, Q::MPIStateArray,
                           t::Real; diffusive=false)
  device = typeof(Q.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  nrealelem = length(topology.realelems)

  Np = dofs_per_element(grid)

  ### update aux variables
  if diffusive
    @launch(device, threads=(Np,), blocks=nrealelem,
            knl_nodal_update_aux!(m, Val(dim), Val(N), f!,
                            Q.data, dg.auxstate.data, dg.diffstate.data, t,
                            topology.realelems))
  else
    @launch(device, threads=(Np,), blocks=nrealelem,
            knl_nodal_update_aux!(m, Val(dim), Val(N), f!,
                            Q.data, dg.auxstate.data, t,
                            topology.realelems))
  end
end

"""
    courant(local_courant::Function, dg::DGModel, m::BalanceLaw,
            Q::MPIStateArray, direction=EveryDirection())
Returns the maximum of the evaluation of the function `local_courant`
pointwise throughout the domain.  The function `local_courant` is given an
approximation of the local node distance `Δx`.  The `direction` controls which
reference directions are considered when computing the minimum node distance
`Δx`.
An example `local_courant` function is
    function local_courant(m::AtmosModel, state::Vars, aux::Vars,
                           diffusive::Vars, Δx)
      return Δt * cmax / Δx
    end
where `Δt` is the time step size and `cmax` is the maximum flow speed in the
model.
"""
function courant(local_courant::Function, dg::DGModel, m::BalanceLaw,
                 Q::MPIStateArray, Δt, direction=EveryDirection())
    grid = dg.grid
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        N = polynomialorder(grid)
        dim = dimensionality(grid)
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        device = grid.vgeo isa Array ? CPU() : CUDA()
        pointwise_courant = similar(grid.vgeo, Nq^dim, nrealelem)
        @launch(device, threads=(Nq, Nq, Nqk), blocks=nrealelem,
        Grids.knl_min_neighbor_distance!(Val(N), Val(dim), direction,
                                         pointwise_courant, grid.vgeo, topology.realelems))
        @launch(device, threads=(Nq*Nq*Nqk,), blocks=nrealelem,
                knl_local_courant!(m, Val(dim), Val(N), pointwise_courant,
                local_courant, Q.data, dg.auxstate.data,
                dg.diffstate.data, topology.realelems, direction, Δt))
        rank_courant_max = maximum(pointwise_courant)
    else
        rank_courant_max = typemin(eltype(Q))
    end

    MPI.Allreduce(rank_courant_max, max, topology.mpicomm)
end

function copy_stack_field_down!(dg::DGModel, m::BalanceLaw,
                                auxstate::MPIStateArray, fldin, fldout)

  device = typeof(auxstate.data) <: Array ? CPU() : CUDA()

  grid = dg.grid
  topology = grid.topology

  dim = dimensionality(grid)
  N = polynomialorder(grid)
  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  # do integrals
  nelem = length(topology.elems)
  nvertelem = topology.stacksize
  nhorzelem = div(nelem, nvertelem)

  @launch(device, threads=(Nq, Nqk, 1), blocks=nhorzelem,
          knl_copy_stack_field_down!(Val(dim), Val(N), Val(nvertelem),
                                     auxstate.data, 1:nhorzelem, Val(fldin),
                                     Val(fldout)))
end

function MPIStateArrays.MPIStateArray(dg::DGModel, commtag=888)
  bl = dg.balancelaw
  grid = dg.grid

  state = create_state(bl, grid, commtag)

  return state
end

