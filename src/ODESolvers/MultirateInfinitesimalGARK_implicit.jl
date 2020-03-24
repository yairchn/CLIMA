export MRIGARKImplicit
export MRIGARKIRK21aSandu

"""
TODO: Document
"""
mutable struct MRIGARKImplicit{T, RT, AT, LT, Nstages, NΓ, FS, Nstages_sq} <:
               AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    slowrhs!
    "rhs linear operator"
    slowrhs_linear!
    "implicit operator, pre-factorized"
    implicitoperator!
    "linear solver"
    linearsolver::LT
    "Storage for RHS during the `MRIGARKImplicit` update"
    Rstages::NTuple{Nstages, AT}
    "Storage for the linear solver rhs vector"
    Qhat::AT
    "RK coefficient matrices for coupling coefficients"
    Γs::NTuple{NΓ, SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}}
    "RK coefficient matrices for embedded scheme"
    γ̂s::NTuple{NΓ, SArray{NTuple{1, Nstages}, RT, 1, Nstages}}
    "RK coefficient vector C (time scaling)"
    Δc::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    "fast solver"
    fastsolver::FS

    function MRIGARKImplicit(
        slowrhs!,
        slowrhs_linear!,
        linearsolver::AbstractLinearSolver,
        fastsolver,
        Γs,
        γ̂s,
        Q::AT,
        dt = nothing,
        t0 = 0,
    ) where {AT <: AbstractArray}

        @assert dt != nothing

        NΓ = length(Γs)
        Nstages = size(Γs[1], 1)
        T = eltype(Q)
        RT = real(T)
        LT = typeof(linearsolver)

        # Compute the Δc coefficients
        Δc = sum(Γs[1], dims = 2)[:]

        # Scale in the Δc to the Γ and γ̂, and convert to real type
        Γs = ntuple(k -> RT.(Γs[k] ./ Δc), NΓ)
        γ̂s = ntuple(k -> RT.(γ̂s[k] / Δc[Nstages]), NΓ)

        # Convert to real type
        Δc = RT.(Δc)

        # create storage for the stage values
        Rstages = ntuple(i -> similar(Q), Nstages)
        Qhat = similar(Q)

        # create implicit operator. Term comes from the following
        # expression in (2.7) from Sandu (2019):
        #
        # v(0) = y_n;  v' = g(v) + f(y_n),  θ ∈ [0, H]; Y^slow_2 = v(H),
        # y_{n+1} = Y^slow_2 - 1/2 g(y_n) + 1/2 f(y_{n+1}) (Implicit term);
        #
        # where 1/2 is the term in the Γ0 matrix (implicit part, 2,2 entry)
        # TODO: Is this right?
        Γ0, = Γs
        α = dt * Γ0[2, 2]
        # Here we are passing NaN for the time since prefactorization assumes the
        # operator is time independent.  If that is not the case the NaN will
        # surface.
        implicitoperator! = prefactorize(
            EulerOperator(slowrhs_linear!, -α),
            linearsolver,
            Q,
            nothing,
            T(NaN),
        )

        FS = typeof(fastsolver)
        new{T, RT, AT, LT, Nstages, NΓ, FS, Nstages^2}(
            RT(dt),
            RT(t0),
            slowrhs!,
            slowrhs_linear!,
            implicitoperator!,
            linearsolver,
            Rstages,
            Qhat,
            Γs,
            γ̂s,
            Δc,
            fastsolver,
        )
    end
end

"""
    MRIGARKIRK21aSandu(f!, fastsolver, Q; dt, t0 = 0)

The 2rd order, 2 stage implicit scheme from Sandu (2019).
"""
function MRIGARKIRK21aSandu(slowrhs!, slowrhs_linear!, linearsolver, fastsolver, Q; dt, t0 = 0)
    T = eltype(Q)
    RT = real(T)
    #! format: off
    Γ0 = [ 1 // 1   0 // 1
          -1 // 2   1 // 2
           0 // 1   0 // 1]
    γ̂0 = [-1 // 2   1 // 2]
    #! format: on
    MRIGARKImplicit(slowrhs!, slowrhs_linear!, linearsolver, fastsolver, (Γ0,), (γ̂0,), Q, dt, t0)
end

# this will only work for iterative solves
# direct solvers use prefactorization
isadjustable(mrigark::MRIGARKImplicit) = mrigark.implicitoperator! isa EulerOperator
function updatedt!(mrigark::MRIGARKImplicit, dt)
    @assert isadjustable(mrigark)
    mrigark.dt = dt
    Γ0 = mrigark.Γs[1]
    α = dt * Γ0[2, 2]
    mrigark.implicitoperator! = EulerOperator(mrigark.slowrhs_linear!, -α)
end
updatetime!(mrigark::MRIGARKImplicit, time) = (mrigark.t = time)

function dostep!(
    Q,
    mrigark::MRIGARKImplicit,
    param,
    timeend::Real,
    adjustfinalstep::Bool,
)
    time, dt = mrigark.t, mrigark.dt
    @assert dt > 0
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end

    dostep!(Q, mrigark, param, time, dt)

    if dt == mrigark.dt
        mrigark.t += dt
    else
        mrigark.t = timeend
    end
    return mrigark.t
end

function dostep!(
    Q,
    mrigark::MRIGARKImplicit,
    param,
    time::Real,
    dt::AbstractFloat,
)
    fast = mrigark.fastsolver

    implicitoperator!, linearsolver = mrigark.implicitoperator!, mrigark.linearsolver
    Rs = mrigark.Rstages
    Δc = mrigark.Δc
    Qhat = mrigark.Qhat

    Nstages = length(Δc)
    groupsize = 256

    slowrhs! = mrigark.slowrhs!
    slowrhs_linear! = mrigark.slowrhs_linear!
    Γs = mrigark.Γs
    NΓ = length(Γs)
    γs = ntuple(k -> ntuple(j -> Γs[k][s, j], s), NΓ)

    ts = time
    for s in 1:div(Nstages, 2)
        # Stage dt
        dts = Δc[2s-1] * dt
        stagetime = ts + dts

        # initialize stage
        slowrhs!(Rs[2s-1], Q, param, ts, increment = false)
        # FIXME: Do we need to call something like `mri_update_rate!`
        # like in MRIGARK_explicit.jl?

        # fast solver
        solve!(Q, mrigark.fastsolver, param; timeend = stagetime)

        # slow solver
        Γ0, = Γs
        α = dt * Γ0[2s, s+1]
        linearoperator! = function (LQ, Q)
            slowrhs_linear!(LQ, Q, p, stagetime; increment = false)
            @. LQ = Q - α * LQ
        end
        # Qhat = Q - ∑_k Γ_{sk} Rs[k]
        mri_create_Qhat!(Qhat, Q, γs, Rs)
        # (Q - α * LQ) = Qhat
        linearsolve!(implicitoperator!, linearsolver, Q, Qhat, p, stagetime)

        # update time
        ts += dts
    end
end

@kernel function mri_create_Qhat!(Qhat, Q, γs, Rs)
    i = @index(Global, Linear)
    @inbounds begin
        NΓ = length(γs)
        Ns = length(γs[1])
        qhat = Q[i]

        for s in 1:Ns
            ri = Rs[s][i]
            sc = γs[1][s]
            qhat -= sc * ri
        end
        Qhat[i] = qhat
    end
end
