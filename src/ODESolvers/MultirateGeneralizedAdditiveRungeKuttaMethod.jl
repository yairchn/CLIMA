export MultirateGeneralizedAdditiveRungeKutta

"""
TODO: Fill out documentation
"""
mutable struct MultirateGeneralizedAdditiveRungeKutta{SS, FS, RT} <: AbstractODESolver
    "slow (outer) solver"
    slow_solver::SS
    "fast (inner) solver"
    fast_solver::FS
    "time step (slow/outer)"
    dt::RT
    "time"
    t::RT

    function MultirateGeneralizedAdditiveRungeKutta(
        slow_solver,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver)
        t0 = slow_solver.t
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
    end
end

function MultirateGeneralizedAdditiveRungeKutta(
    solvers::Tuple,
    Q = nothing;
    dt = getdt(solvers[1]),
    t0 = solvers[1].t,
) where {AT <: AbstractArray}
    if length(solvers) < 2
        error("Must specify atleast two solvers")
    elseif length(solvers) == 2
        fast_solver = solvers[2]
    else
        fast_solver = MultirateGeneralizedAdditiveRungeKutta(solvers[2:end], Q; dt = dt, t0 = t0)
    end

    slow_solver = solvers[1]

    return MultirateGeneralizedAdditiveRungeKutta(slow_solver, fast_solver, Q; dt = dt, t0 = t0)
end

function dostep!(
    Q,
    mrgark::MultirateGeneralizedAdditiveRungeKutta,
    param,
    timeend::Real,
    adjustfinalstep::Bool,
)
    time = mrgark.t
    dt = mrgark.dt

    @assert dt > 0
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end

    dostep!(Q, mrgark, param, time, dt)

    if dt == mrrk.dt
        mrgark.t += dt
    else
        mrgark.t = timeend
    end
    return mrgark.t
end

function dostep!(
    Q,
    mrgark::MultirateGeneralizedAdditiveRungeKutta{SS},
    param,
    time::Real,
    dt::AbstractFloat,
    in_slow_δ = nothing,
    in_slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
) where {SS <: AbstractAdditiveRungeKutta}
    slow = mrgark.slow_solver
    fast = mrgark.fast_solver
end

    implicitoperator!, linearsolver = slow.implicitoperator!, slow.linearsolver
    RKA_explicit, RKA_implicit = slow.RKA_explicit, slow.RKA_implicit
    RKB, RKC = slow.RKB, slow.RKC
    rhs!, rhs_linear! = slow.rhs!, slow.rhs_linear!
    Qstages, Rstages = slow.Qstages, slow.Rstages
    Qhat = slow.Qhat
    split_nonlinear_linear = slow.split_nonlinear_linear
    Qtt = slow.variant_storage.Qtt

    rv_Q = realview(Q)
    rv_Qstages = realview.(Qstages)
    rv_Rstages = realview.(Rstages)
    rv_Qhat = realview(Qhat)
    rv_Qtt = realview(Qtt)

    Nouterstages = length(RKB)
    groupsize = 256

    # begin outer IMEX loop
    for outer_stage in 1:Nouterstages

        # calculate the rhs at first stage to initialize the stage loop
        if outer_stage == 1
            rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)

            # dt has changed, need to rebuild linear operator
            if dt != slow.dt
                α = dt * RKA_implicit[2, 2]
                implicitoperator! = EulerOperator(rhs_linear!, -α)
            end
        end

        # Current outer stage time
        outer_stage_time = time + RKC[outer_stage] * dt

        # Fractional time for inner solver
        if outer_stage == Nouterstages
            γ = 1 - RKC[outer_stage]
        else
            γ = RKC[outer_stage + 1] - RKC[outer_stage]
        end

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        slow_δ = RKB[outer_stage] / (γ)

        # RKB for the slow with fractional time factor remove (since full
        # integration of fast will result in scaling by γ)
        nsubsteps = getdt(fast) > 0 ? ceil(Int, γ * dt / getdt(fast)) : 1
        fast_dt = γ * dt / nsubsteps

        # begin inner loop
        for substep in 1:nsubsteps
            slow_rka = nothing
            if substep == nsubsteps
                slow_rka = RKA[outer_stage % Nouterstages + 1]
            end
            fast_time = outer_stage_time + (substep - 1) * fast_dt
            # want to update Qstages (not Q; this happens after each slow dt)
            dostep!(
                Qstages[outer_stage],
                fast,
                param,
                fast_time,
                fast_dt,
                slow_δ,
                slow_rv_dQ,
                slow_rka,
            )
        end

        # this kernel initializes Qtt for the linear solver
        event = Event(device(Q))
        event = stage_update!(device(Q), groupsize)(
            variant,
            rv_Q,
            rv_Qstages,
            rv_Rstages,
            rv_Qhat,
            rv_Qtt,
            RKA_explicit,
            RKA_implicit,
            dt,
            Val(outer_stage),
            Val(split_nonlinear_linear),
            slow_δ,
            slow_rv_dQ;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(device(Q), event)

        # solves Q_tt = Qhat + dt * RKA_implicit[outer_stage, outer_stage] * rhs_linear!(Q_tt)
        linearsolve!(implicitoperator!, linearsolver, Qtt, Qhat, p, stagetime)

        # update Qstages
        Qstages[outer_stage] .+= Qtt

        rhs!(Rstages[istage], Qstages[outer_stage], p, stagetime, increment = false)
    end

    # TODO: Safe to combine this with outer loop above?
    if split_nonlinear_linear
        for istage in 1:Nouterstages
            stagetime = time + RKC[istage] * dt
            rhs_linear!(
                Rstages[istage],
                Qstages[istage],
                p,
                stagetime,
                increment = true,
            )
        end
    end

    # compose the final solution
    event = Event(device(Q))
    event = solution_update!(device(Q), groupsize)(
        variant,
        rv_Q,
        rv_Rstages,
        RKB,
        dt,
        Val(Nstages),
        slow_δ,
        slow_rv_dQ,
        slow_scaling;
        ndrange = length(rv_Q),
        dependencies = (event,),
    )
    wait(device(Q), event)
end
