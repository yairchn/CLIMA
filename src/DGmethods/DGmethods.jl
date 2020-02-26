module DGmethods

using MPI
using ..MPIStateArrays
using ..MPIStateArrays.CMBuffers: friendlysynchronize
using ..Mesh.Grids
using ..Mesh.Topologies
using StaticArrays
using ..SpaceMethods
using ..VariableTemplates
using DocStringExtensions
using GPUifyLoops
using CUDAdrv: CuEvent, CuStream, CuDefaultStream

export BalanceLaw, DGModel, init_ode_state

include("balancelaw.jl")
include("DGmodel.jl")
include("NumericalFluxes.jl")
include("DGmodel_kernels.jl")

end
