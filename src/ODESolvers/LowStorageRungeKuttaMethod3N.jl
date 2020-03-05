export LowStorageRungeKutta3N
export LSRK3N184ParsaniKetchesonDeconinck

include("LowStorageRungeKuttaMethod3N_kernels.jl")

"""
    LowStorageRungeKutta3N(f, γ1, γ2, γ3, δ, β, δ, c, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a low-storage Runge-Kutta scheme using 3N
storage based on the provided `γ1`, `γ2`, `γ3`, `β`, and `c` coefficient arrays.

The available concrete implementations are:

"""
mutable struct LowStorageRungeKutta3N{T, RT, AT, Nstages} <: AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  "rhs function"
  rhs!
  "Storage for solution scratch array 2 during the LowStorageRungeKutta update"
  Q2::AT
  "Storage for solution scratch array 3 during the LowStorageRungeKutta update"
  Q3::AT
  "Storage for RHS during the LowStorageRungeKutta update"
  dQ::AT
  "low storage RK coefficient vector γ1 (combine stages)"
  γ1::NTuple{Nstages, RT}
  "low storage RK coefficient vector γ2 (combine stages)"
  γ2::NTuple{Nstages, RT}
  "low storage RK coefficient vector γ3 (combine stages)"
  γ3::NTuple{Nstages, RT}
  "low storage RK coefficient vector β (scaling RHS)"
  β::NTuple{Nstages, RT}
  "low storage RK coefficient vector δ (updating Q2 scratch)"
  δ::NTuple{Nstages, RT}
  "low storage RK coefficient vector c (time update)"
  c::NTuple{Nstages, RT}

  function LowStorageRungeKutta3N(rhs!, γ1, γ2, γ3, β, δ, c, Q::AT;
                                  dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)

    Q2 = similar(Q)
    Q3 = similar(Q)
    dQ = similar(Q)
    fill!(dQ, 0)

    NS = length(γ1)

    new{T, RT, AT, NS}(RT(dt), RT(t0), rhs!, Q2, Q3, dQ, γ1, γ2, γ3, β, δ, c)
  end
end

"""
    LowStorageRungeKutta3N(spacedisc::AbstractSpaceMethod, x...; kw...)

Defines the ODE right-hand side function from the space method 'spacedisc` then
calls the default constructor for `LowStorageRungeKutta3N`
"""
function LowStorageRungeKutta3N(spacedisc::AbstractSpaceMethod, args...; kws...)
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x...,
                                                   increment = increment)
  LowStorageRungeKutta3N(rhs!, args...; kws...)
end

updatedt!(lsrk::LowStorageRungeKutta3N, dt) = (lsrk.dt = dt)
updatetime!(lsrk::LowStorageRungeKutta3N, time) = (lsrk.t = time)

"""
    dostep!(Q, lsrk::LowStorageRungeKutta3N, p, timeend::Real,
            adjustfinalstep::Bool)

Use the 3N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time, to the time `timeend`. If `adjustfinalstep == true` then
`dt` is adjusted so that the step does not take the solution beyond the
`timeend`.
"""
function dostep!(Q, lsrk::LowStorageRungeKutta3N, p, timeend::Real,
                 adjustfinalstep::Bool)
  time, dt = lsrk.t, lsrk.dt
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
  end
  @assert dt > 0

  dostep!(Q, lsrk, p, time, dt)

  if dt == lsrk.dt
    lsrk.t += dt
  else
    lsrk.t = timeend
  end

end


"""
    dostep!(Q, lsrk::LowStorageRungeKutta3N, p, time::Real, dt::Real)

Use the 3N low storage Runge--Kutta method `lsrk` to step `Q` forward in time
from the current time `time` to final time `time + dt`.
"""
function dostep!(Q1, lsrk::LowStorageRungeKutta3N, p, time::Real, dt::Real)
  γ1, γ2, γ3 = lsrk.γ1, lsrk.γ2, lsrk.γ3
  β, δ, c = lsrk.β, lsrk.δ, lsrk.c
  rhs!, dQ = lsrk.rhs!, lsrk.dQ
  Q2, Q3 = lsrk.Q2, lsrk.Q3

  rv_Q1 = realview(Q1)
  rv_Q2 = realview(Q2)
  rv_Q3 = realview(Q3)
  rv_dQ = realview(dQ)

  threads = 256
  blocks = div(length(rv_Q1) + threads - 1, threads)

  for s = 1:length(γ1)
    rhs!(dQ, Q1, p, time + c[s] * dt, increment = false)

    @launch(device(Q1), threads=threads, blocks=blocks,
            update3N!(rv_dQ, rv_Q1, rv_Q2, rv_Q3,
                      γ1[s], γ2[s], γ3[s], β[s], δ[s], dt, s == 1))
  end
end

"""
    LSRK3N184ParsaniKetchesonDeconinck(f, Q; dt, t0)

This function returns a [`LowStorageRungeKutta3N`](@ref) time stepping object
for explicitly time stepping the differential equation given by the
right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses the fourth-order, 18-stage, 3-register, Runge--Kutta scheme of
Parsani, Ketcheson, and Deconinck(2012) with optimized stability region.

### References

    @article{parsani2013optimized,
      title={Optimized explicit Runge--Kutta schemes for the spectral difference
              method applied to wave propagation problems},
      author={Parsani, Matteo and Ketcheson, David I and Deconinck, W},
      journal={SIAM Journal on Scientific Computing},
      volume={35},
      number={2},
      pages={A957--A986},
      year={2013},
      publisher={SIAM},
      doi={10.1137/120885899}
    }
"""
function LSRK3N184ParsaniKetchesonDeconinck(f, Q::AT; dt=0,
                                            t0=0) where {AT <: AbstractArray}
  T = eltype(Q)
  RT = real(T)
  c = (RT( 0.0000000000000000e+00),
       RT( 1.2384169480626298e-01),
       RT( 1.1574324659554065e+00),
       RT( 5.4372099141546926e-01),
       RT( 8.8394666834280744e-01),
       RT(-1.2212042176605774e-01),
       RT( 4.4125685133082082e-01),
       RT( 3.8039092095473748e-01),
       RT( 5.4591107347528367e-02),
       RT( 4.8731855535356028e-01),
       RT(-2.3007964303896034e-01),
       RT(-1.8907656662915873e-01),
       RT( 8.1059805668623763e-01),
       RT( 7.7080875997868803e-01),
       RT( 1.1712158507200179e+00),
       RT( 1.2755351018003545e+00),
       RT( 8.0422507946168564e-01),
       RT( 9.7508680250761848e-01))

  β = (RT( 1.2384169480626298e-01),
       RT( 1.0176262534280349e+00),
       RT(-6.9732026387527429e-02),
       RT( 3.4239356067806476e-01),
       RT( 1.8177707207807942e-02),
       RT(-6.1188746289480445e-03),
       RT( 7.8242308902580354e-02),
       RT(-3.7642864750532951e-01),
       RT(-4.5078383666690258e-02),
       RT(-7.5734228201432585e-01),
       RT(-2.7149222760935121e-01),
       RT( 1.1833684341657344e-03),
       RT( 2.8858319979308041e-02),
       RT( 4.6005267586974657e-01),
       RT( 1.8014887068775631e-02),
       RT(-1.5508175395461857e-02),
       RT(-4.0095737929274988e-01),
       RT( 1.4949678367038011e-01))

  γ1 = (RT( 0.0000000000000000e+00),
        RT( 1.1750819811951678e+00),
        RT( 3.0909017892654811e-01),
        RT( 1.4409117788115862e+00),
        RT(-4.3563049445694069e-01),
        RT( 2.0341503014683893e-01),
        RT( 4.9828356971917692e-01),
        RT( 3.5307737157745489e+00),
        RT(-7.9318790975894626e-01),
        RT( 8.9120513355345166e-01),
        RT( 5.7091009196320974e-01),
        RT( 1.6912188575015419e-02),
        RT( 1.0077912519329719e+00),
        RT(-6.8532953752099512e-01),
        RT( 1.0488165551884063e+00),
        RT( 8.3647761371829943e-01),
        RT( 1.3087909830445710e+00),
        RT( 9.0419681700177323e-01))

  γ2 = (RT( 1.0000000000000000e+00),
        RT(-1.2891068509748144e-01),
        RT( 3.5609406666728954e-01),
        RT(-4.0648075226104241e-01),
        RT( 6.0714786995207426e-01),
        RT( 1.0253501186236846e+00),
        RT( 2.4411240760769423e-01),
        RT(-1.2813606970134104e+00),
        RT( 8.1625711892373898e-01),
        RT( 1.0171269354643386e-01),
        RT( 1.9379378662711269e-01),
        RT( 7.4408643544851782e-01),
        RT(-1.2591764563430008e-01),
        RT( 1.1996463179654226e+00),
        RT( 4.5772068865370406e-02),
        RT( 8.3622292077033844e-01),
        RT(-1.4179124272450148e+00),
        RT( 1.3661459065331649e-01))

  γ3 = (RT( 0.0000000000000000e+00),
        RT( 0.0000000000000000e+00),
        RT( 0.0000000000000000e+00),
        RT( 2.5583378537249163e-01),
        RT( 5.2676794366988289e-01),
        RT(-2.5648375621792202e-01),
        RT( 3.1932438003236391e-01),
        RT(-3.1106815010852862e-01),
        RT( 4.7631196164025996e-01),
        RT(-9.8853727938895783e-02),
        RT( 1.9274726276883622e-01),
        RT( 3.2389860855971508e-02),
        RT( 7.5923980038397509e-02),
        RT( 2.0635456088664017e-01),
        RT(-8.9741032556032857e-02),
        RT( 2.6899932505676190e-02),
        RT( 4.1882069379552307e-02),
        RT( 6.2016148912381761e-02))

  δ = (RT( 1.0000000000000000e+00),
       RT( 3.5816500441970289e-01),
       RT( 5.8208024465093577e-01),
       RT(-2.2615285894283538e-01),
       RT(-2.1715466578266213e-01),
       RT(-4.6990441450888265e-01),
       RT(-2.7986911594744995e-01),
       RT( 9.8513926355272197e-01),
       RT(-1.1899324232814899e-01),
       RT( 4.2821073124370562e-01),
       RT(-8.2196355299900403e-01),
       RT( 5.8113997057675074e-02),
       RT(-6.1283024325436919e-01),
       RT( 5.6800136190634054e-01),
       RT(-3.3874970570335106e-01),
       RT(-7.3071238125137772e-01),
       RT( 8.3936016960374532e-02),
       RT( 0.0000000000000000e+00))

  LowStorageRungeKutta3N(f, γ1, γ2, γ3, β, δ, c, Q; dt=dt, t0=t0)
end
