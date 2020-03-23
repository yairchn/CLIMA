export ExplicitRungeKutta
export ERK43BogackiShampine

include("ExplicitRungeKuttaMethod_kernels.jl")

mutable struct ExplicitRungeKutta{T, RT, AT, EC, Nstages, Nstages_sq} <:
               AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    rhs!
    "Storage for the RHS evaluations"
    Rstages::NTuple{Nstages, AT}
    "Storage for the stages"
    Qstages::NTuple{Nstages, AT}
    "Storage for the error estimate"
    δ::Union{Nothing, AT}
    "RK coefficient vector A (rhs scaling)"
    RKA::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient vector B (rhs add in scaling)"
    RKB::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector B for the embedded scheme"
    RKB_embedded::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector C (time scaling)"
    RKC::SArray{Tuple{Nstages}, RT, 1, Nstages}
    error_controller::EC

    function ExplicitRungeKutta(
        rhs!,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q::AT;
        dt = 0,
        t0 = 0,
        error_controller = NoController()
    ) where {AT <: AbstractArray}
        T = eltype(Q)
        RT = real(T)
        EC = typeof(error_controller)
        Nstages = length(RKB)
        
        δ = error_controller isa NoController ? nothing : similar(Q)
        Rstages = ntuple(i -> similar(Q), Nstages)
        Qstages = ntuple(i -> similar(Q), Nstages)

        new{T, RT, AT, EC, Nstages, Nstages ^ 2}(
            RT(dt),
            RT(t0),
            rhs!,
            Rstages,
            Qstages,
            δ,
            RKA,
            RKB,
            RKB_embedded,
            RKC,
            error_controller
        )
    end
end

updatedt!(erk::ExplicitRungeKutta, dt) = erk.dt = dt

function ExplictRungeKutta(
    spacedisc::AbstractSpaceMethod,
    RKA,
    RKB,
    RKC,
    Q::AT;
    dt = 0,
    t0 = 0,
    error_controller = NoController()
) where {AT <: AbstractArray}
    rhs! =
        (x...; increment) ->
            SpaceMethods.odefun!(spacedisc, x..., increment = increment)
    ExplicitRungeKutta(
        rhs!,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q;
        dt = dt,
        t0 = t0,
        error_controller = error_controller
    )
end

updatedt!(erk::ExplicitRungeKutta, dt) = (erk.dt = dt)
updatetime!(erk::ExplicitRungeKutta, time) = (erk.t = time)

function dostep!(
    (Qnp1, Qn, error_estimate),
    erk::ExplicitRungeKutta,
    p,
    time,
    dt,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    RKA, RKB, RKC = erk.RKA, erk.RKB, erk.RKC
    RKB_embedded = erk.RKB_embedded
    Nstages = length(RKB)
    rhs! = erk.rhs!
    Rstages, Qstages, δ = erk.Rstages, erk.Qstages, erk.δ

    rv_error_estimate = realview(error_estimate)
    rv_Qn = realview(Qn)
    rv_Qnp1 = realview(Qnp1)
    rv_Rstages = realview.(Rstages)
    rv_Qstages = realview.(Qstages)
    groupsize = 256

    for s in 1:Nstages
        event = Event(device(Qn))
        event = stage_update!(device(Qn), groupsize)(
            rv_Qn,
            rv_Qstages,
            rv_Rstages,
            RKA,
            dt,
            Val(s),
            slow_δ,
            slow_rv_dQ,
            ndrange = length(rv_Qn),
            dependencies = (event,),
        )
        wait(device(Qn), event)
        rhs!(Rstages[s], Qstages[s], p, time + RKC[s] * dt, increment = false)
    end

    event = Event(device(Qn))
    event = solution_update!(device(Qn), groupsize)(
        rv_Qnp1,
        rv_Qn,
        rv_error_estimate,
        rv_Qstages,
        rv_Rstages,
        RKB,
        RKB_embedded,
        dt,
        Val(Nstages),
        slow_δ,
        slow_rv_dQ,
        slow_scaling,
        ndrange = length(rv_Qn),
        dependencies = (event,),
    )
    wait(device(Qn), event)
end

function ERK43BogackiShampine(F, Q::AT; dt = 0, t0 = 0) where {AT <: AbstractArray}
    RKA = [0       0       0       0; 
           1 // 2  0       0       0;
           0       3 // 4  0       0;
           2 // 9  1 // 3  4 // 9  0]

    RKB = [2 // 9, 1 // 3, 4 // 9, 0]
    RKB_embedded = [7 // 24, 1 // 4, 1 // 3, 1 // 8]
    RKC = [0, 1 // 2, 3 // 4, 1]

    ExplicitRungeKutta(F, RKA, RKB, RKB_embedded, RKC, Q; dt = dt, t0 = t0)
end

