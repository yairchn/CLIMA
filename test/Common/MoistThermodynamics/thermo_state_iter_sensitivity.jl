using Test
using CLIMA.MoistThermodynamics

using CLIMA.PlanetParameters
using CLIMA.RootSolvers
using LinearAlgebra
using Plots

function plot_prop(ts, dir, filename, prop)
  TS = last.(ts)
  plot(1:length(getproperty(TS[1],prop)), getproperty(TS[1],prop))
  for i in 2:length(TS)
    plot!(1:length(getproperty(TS[i],prop)), getproperty(TS[i],prop))
  end
  png(joinpath(dir,filename))
end


@testset "moist thermodynamics - thermo state convergence sensitivity" begin
  FT = Float64
  n = 100
  ΔT_err_max = 0
  err_max = 0
  sat_adjust_call_count = 0
  ρ_range  = range(1e-3, stop=1.6,  length=n);
  RH_range = range(0, stop=1,       length=n);
  T_range  = range(T_min, stop=400, length=n);
  tol = FT(1e-2)
  maxiter_range = 1
  domain_dim = (length(RH_range), length(ρ_range), length(T_range), length(maxiter_range));
  TS = Array{ThermodynamicState}(undef, domain_dim);
  SOL = Array{RootSolvers.VerboseSolutionResults}(undef, domain_dim);
  results = zeros(Float64, length(maxiter_range));
  maxiter_range = maxiter_range isa Array ? maxiter_range : [maxiter_range,]

  mkpath("output")
  mkpath(joinpath("output","MoistThermoAnalysis"))
  dir = joinpath("output","MoistThermoAnalysis")

  for method in (PhaseEquil_NewtonMethod, PhaseEquil)
    for (p, maxiter) in enumerate(maxiter_range)
    for (i,RH) in enumerate(RH_range)
    for (j,ρ) in enumerate(ρ_range)
    for (k,T) in enumerate(T_range)
      q_sat = q_vap_saturation(T, ρ)
      q_tot = min(RH*q_sat, 1)
      q_pt = PhasePartition_equil(T, ρ, q_tot)
      e_int = internal_energy(T, q_pt)
      ts_eq, sol, sa_called = method(e_int, q_tot, ρ, maxiter, tol)
      ΔT_err_max = max(abs(T - air_temperature(ts_eq)), ΔT_err_max)
      err_max = max(abs(sol.err),err_max)
      sat_adjust_call_count += sa_called
      TS[i,j,k,p] = ts_eq
      SOL[i,j,k,p] = sol
    end
    end
    end
    end

    q_sat_range = q_vap_saturation.(T_range, ρ_range)
    q_tot_range = RH_range .* q_sat_range
    e_int_range = internal_energy.(T_range, PhasePartition.(q_tot_range,Ref(FT(0)),q_tot_range))

    SOL_max_iter = SOL[:,:,:,end]
    println("------------------------------- Pass/fail rate for $(method)")
    @show length(SOL_max_iter)
    @show sum(getproperty.(SOL_max_iter, :converged))
    @show sum(getproperty.(SOL_max_iter, :converged))/length(SOL_max_iter)
    @show 1-sum(getproperty.(SOL_max_iter, :converged))/length(SOL_max_iter)
    @show ΔT_err_max
    @show err_max
    @show sat_adjust_call_count
    @show sat_adjust_call_count/length(SOL)

    # results = [count(map(x->x.converged, SOL[:,:,:,p]))/length(SOL[:,:,:,1]) for (p, maxiter) in enumerate(maxiter_range)]
    # plot(maxiter_range, results, xlabel="maxiter", ylabel="% converged"); png(joinpath(dir,"converged_percent"))

    # results = [sum(map(x->sum(abs.(x.err)), SOL[:,:,:,p]))/length(SOL[:,:,:,1]) for (p, maxiter) in enumerate(maxiter_range)]
    # plot(maxiter_range, results, xlabel="maxiter", ylabel="error"); png(joinpath(dir,"err_vs_maxiter"))

    # results = [sum(map(x->sum(abs.(x.iter_performed)), SOL[:,:,:,p]))/length(SOL[:,:,:,1]) for (p, maxiter) in enumerate(maxiter_range)]
    # plot(maxiter_range, results, xlabel="maxiter", ylabel="iter_performed"); png(joinpath(dir,"iter_performed"))

    if length(SOL_max_iter)<2*10^3
      IDX_nc = findall(map(x->!x.converged, SOL[:,:,:,end]))
      IDX_co = findall(map(x->x.converged, SOL[:,:,:,end]))

      ts_nc  = zip(TS[IDX_nc],SOL[IDX_nc])
      ts_co  = zip(TS[IDX_co],SOL[IDX_co])

      plot(1:length(ts_nc), map(ts->ts.T, first.(ts_nc)), xlabel="case, no particular order", ylabel="temperature", title="non-converged"); png(joinpath(dir,"T_non_converged_$(method)"))
      plot(1:length(ts_co), map(ts->ts.T, first.(ts_co)), xlabel="case, no particular order", ylabel="temperature", title="converged"    ); png(joinpath(dir,"T_converged_$(method)"))

      if !isempty(ts_nc)
        plot_prop(ts_nc, dir, "non_converged_root_histories_$(method)", :root_history)
        plot_prop(ts_nc, dir, "non_converged_err_histories_$(method)", :err_history)
      end
      plot_prop(ts_co, dir, "converged_root_histories_$(method)", :root_history)
      plot_prop(ts_co, dir, "converged_err_histories_$(method)", :err_history)
    end

  end


end
