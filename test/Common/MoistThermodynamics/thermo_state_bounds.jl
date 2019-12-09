using Pkg

using Test
using CLIMA.MoistThermodynamics

using CLIMA.PlanetParameters
using CLIMA.RootSolvers
using LinearAlgebra
using Plots

@testset "moist thermodynamics - bounds" begin

  FT = Float64
  q_tot_range = range(0, stop=10^(-1), step=10^(-2))
  ρ_range     = range(1e-3, stop=1.6, step=10^(-1))
  e_int_range = range(-1e5, stop=1e5, step=10^(3))
  q_tot_dim = length(q_tot_range)
  ρ_dim     = length(ρ_range)
  e_int_dim = length(e_int_range)
  @show q_tot_dim, ρ_dim, e_int_dim

  TS = Array{ThermodynamicState}(undef, (q_tot_dim, ρ_dim, e_int_dim))
  SOL = Array{RootSolvers.VerboseSolutionResults}(undef, (q_tot_dim, ρ_dim, e_int_dim))
  maxiter = 10
  tol = 1e-3

  for (i,q_tot) in enumerate(q_tot_range)
  for (j,ρ) in enumerate(ρ_range)
  for (k,e_int) in enumerate(e_int_range)
    ts_eq, sol  = PhaseEquil(e_int, q_tot, ρ, maxiter, tol)
    ts_eq, sol  = PhaseEquil_NewtonMethod(e_int, q_tot, ρ, maxiter, tol)
    TS[i,j,k] = ts_eq
    SOL[i,j,k] = sol
  end
  end
  end

# function all_phases(T, P, n_points, q)
#   M = zeros(n_points, n_points)
#   for (i, t) in enumerate(T)
#     for (j, p) in enumerate(P)
#       M[i,j] = phase(t, p, q...)
#     end
#   end
#   return M
# end
# M = all_phases(T, P, n_points, q);
# n_ice = count(x->x==1, M)
# n_liq = count(x->x==2, M)
# n_gas = count(x->x==3, M)
# contourf(T, P, (x, y)->phase(x, y, q), color=:viridis, xlabel="T", ylabel="P")
# contourf(q_tot_range, e_int_range, (x, y)->phase(x, y, q), color=:viridis, xlabel="T", ylabel="P")

  mkpath("output")
  mkpath(joinpath("output","MoistThermoAnalysis"))
  dir = joinpath("output","MoistThermoAnalysis")

  contourf(e_int_range, ρ_range    , map(x->x.converged, SOL[1,:,:]), color=:viridis, xlabel="e_int", ylabel="ρ")
  savefig(joinpath(dir,"converged1.png"))
  contourf(q_tot_range, e_int_range, map(x->x.converged, SOL[:,1,:]), color=:viridis, xlabel="q_tot", ylabel="e_int")
  savefig(joinpath(dir,"converged2.png"))
  contourf(q_tot_range, ρ_range    , map(x->x.converged, SOL[:,:,1]), color=:viridis, xlabel="q_tot", ylabel="ρ")
  savefig(joinpath(dir,"converged3.png"))

  fun(ts) = air_pressure(ts)
  fun(ts) = air_temperature(ts)
  fun(ts) = PhasePartition(ts).liq

  contourf(e_int_range, ρ_range    , map(x->fun(x), TS[1,:,:]), color=:viridis, xlabel="e_int", ylabel="ρ")
  contourf(q_tot_range, e_int_range, map(x->fun(x), TS[:,1,:]), color=:viridis, xlabel="q_tot", ylabel="e_int")
  contourf(q_tot_range, ρ_range    , map(x->fun(x), TS[:,:,1]), color=:viridis, xlabel="q_tot", ylabel="ρ")


end
