using Test
using CLIMA
using CLIMA.ODESolvers
using CLIMA.LinearSolvers
using StaticArrays
using LinearAlgebra

CLIMA.init()
const ArrayType = CLIMA.array_type()

const explicit_methods = [
                          (ERK43BogackiShampine, 3)
                         ]

@testset "1-rate ODE" begin
  function rhs!(dQ, Q, ::Nothing, time; increment)
    if increment
      dQ .+= Q * cos(time)
    else
      dQ .= Q * cos(time)
    end
  end
  exactsolution(q0, time) = q0 * exp(sin(time))

  @testset "Explicit methods convergence" begin
    finaltime = 20.0
    dts = [2.0 ^ (-k) for k = 0:7]
    errors = similar(dts)
    q0 = ArrayType === Array ? [1.0] : range(-1.0, 1.0, length = 303)
    for (method, expected_order) in explicit_methods
      for (n, dt) in enumerate(dts)
        Q = ArrayType(q0)
        solver = method(rhs!, Q; dt = dt, t0 = 0.0)
        #solver = ErrorAdaptiveSolver(solver, NoController(), Q)
        solver = ErrorAdaptiveSolver(solver, IntegralController(0.9, 1e-5), Q)
        solve!(Q, solver; timeend = finaltime)
        Q = Array(Q)
        errors[n] = maximum(@. abs(Q - exactsolution(q0, finaltime)))
      end
      rates = log2.(errors[1:end-1] ./ errors[2:end])
      @show errors
      @show rates
      @test isapprox(rates[end], expected_order; atol = 0.17)
    end
  end

  #@testset "Explicit methods composition of solve!" begin
  #  halftime = 10.0
  #  finaltime = 20.0
  #  dt = 0.75
  #  for (method, _) in explicit_methods
  #    q0 = ArrayType === Array ? [1.0] : range(1.0, 2.0, length = 303)
  #    Q1 = ArrayType(q0)
  #    solver1 = method(rhs!, Q1; dt = dt, t0 = 0.0)
  #    solve!(Q1, solver1; timeend = finaltime)

  #    Q2 = ArrayType(q0)
  #    solver2 = method(rhs!, Q2; dt = dt, t0 = 0.0)
  #    solve!(Q2, solver2; timeend = halftime, adjustfinalstep = false)
  #    solve!(Q2, solver2; timeend = finaltime)

  #    @test Array(Q2) == Array(Q1)
  #  end
  #end
end
