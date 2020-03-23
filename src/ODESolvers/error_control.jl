export ErrorAdaptiveSolver, NoController, IntegralController

using LinearAlgebra: norm

struct ErrorAdaptiveSolver{S, AT, EC} <: AbstractODESolver
  solver::S
  candidate::AT
  error_estimate::AT
  error_controller::EC
end

function ErrorAdaptiveSolver(solver, error_controller, Q)
  AT = typeof(Q)
  candidate = similar(Q)
  error_estimate = similar(Q)
  ErrorAdaptiveSolver(solver, candidate, error_estimate, error_controller)
end

gettime(eas::ErrorAdaptiveSolver) = gettime(eas.solver)
getdt(eas::ErrorAdaptiveSolver) = getdt(eas.solver)
updatedt!(eas::ErrorAdaptiveSolver, dt) = updatedt!(eas.solver, dt)
updatetime!(eas::ErrorAdaptiveSolver, dt) = updatetime!(eas.solver, dt)

function dostep!(
    Q,
    eas::ErrorAdaptiveSolver,
    p,
    timeend::Real;
    adjustfinalstep::Bool
)
    candidate = eas.candidate
    error_estimate = eas.error_estimate
    time = gettime(eas)
  
    acceptstep = false
    while !acceptstep
      dt = getdt(eas)
     
      dostep!((candidate, Q, error_estimate), eas.solver, p, time, dt)
      acceptstep, newdt = eas.error_controller(dt, error_estimate)
      !acceptstep && updatedt!(eas, newdt)
    end
    
    dt = getdt(eas)
    if adjustfinalstep && time + dt > timeend
      dt = timeend - time
      updatedt!(eas, dt)
      dostep!((candidate, Q, error_estimate), eas.solver, p, time, dt)
    end

    Q .= candidate

    #eas.solver.t += dt
    #eas.solver.t
    updatetime!(eas, time + dt)
    return gettime(eas)
end


abstract type AbstractErrorController end

struct NoController <: AbstractErrorController end
function (obj::NoController)(dt, δ)
  return true, dt
end

struct IntegralController{FT} <: AbstractErrorController
  safety_factor::FT
  tolerance::FT
end

function (obj::IntegralController)(dt, δ)
  p = 2
  norm_δ = norm(δ, Inf, false)
  e1 = norm_δ / obj.tolerance
  newdt = obj.safety_factor * dt * (1 / e1) ^ (1 / p)
  return e1 <= 1, newdt
end

