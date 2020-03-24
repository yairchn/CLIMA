using Test
using CLIMA
using CLIMA.ODESolvers
using CLIMA.LinearSolvers
using StaticArrays
using LinearAlgebra

CLIMA.init()
const ArrayType = CLIMA.array_type()

const mrigark_methods = [
    (MRIGARKIRK21aSandu, 2)
]

const fast_mrigark_methods = [
    (LSRK144NiegemannDiehlBusch, 4)
]

struct DivideLinearSolver <: AbstractLinearSolver end
function LinearSolvers.prefactorize(
    linearoperator!,
    ::DivideLinearSolver,
    args...,
)
    linearoperator!
end
function LinearSolvers.linearsolve!(
    linearoperator!,
    ::DivideLinearSolver,
    Qtt,
    Qhat,
    args...,
)
    @. Qhat = 1 / Qhat
    linearoperator!(Qtt, Qhat, args...)
    @. Qtt = 1 / Qtt
end


@testset "Implicit MRI GARK test" begin
    @testset "2-rate problem" begin
        ω = 100
        λf = -10
        λs = -1
        ξ = 1 // 10
        α = 1
        ηfs = ((1 - ξ) / α) * (λf - λs)
        ηsf = -ξ * α * (λf - λs)
        Ω = @SMatrix [
            λf ηfs
            ηsf λs
        ]

        function rhs_fast!(dQ, Q, param, t; increment)
            @inbounds begin
                increment || (dQ .= 0)
                yf = Q[1]
                ys = Q[2]
                gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                gs = (-2 + ys^2 - cos(t)) / 2ys
                dQ[1] += Ω[1, 1] * gf + Ω[1, 2] * gs - ω * sin(ω * t) / 2yf
            end
        end

        function rhs_slow!(dQ, Q, param, t; increment)
            @inbounds begin
                increment || (dQ .= 0)
                yf = Q[1]
                ys = Q[2]
                gf = (-3 + yf^2 - cos(ω * t)) / 2yf
                gs = (-2 + ys^2 - cos(t)) / 2ys
                dQ[2] += Ω[2, 1] * gf + Ω[2, 2] * gs - sin(t) / 2ys
            end
        end

        exactsolution(t) = [sqrt(3 + cos(ω * t)); sqrt(2 + cos(t))]

        # TODO: Do we still need to pass a separate RHS for the linear part?
        function rhs_zero!(dQ, Q, param, t; increment)
            if !increment
                dQ .= 0
            end
        end

        finaltime = 5π / 2
        dts = [2.0^(-k) for k in 2:9]
        error = similar(dts)
        for (slow_method, slow_expected_order) in mrigark_methods
            for (fast_method, fast_expected_order) in fast_mrigark_methods
                for (n, fast_dt) in enumerate(dts)
                    Q = exactsolution(0)
                    slow_dt = ω * fast_dt
                    fast_solver = fast_method(rhs_fast!, Q; dt = fast_dt)
                    solver = slow_method(rhs_slow!, rhs_zero!,
                                         DivideLinearSolver(),
                                         fast_solver, Q; dt = slow_dt)
                    solve!(Q, solver; timeend = finaltime)
                    error[n] = norm(Q - exactsolution(finaltime))
                end

                rate = log2.(error[1:(end - 1)] ./ error[2:end])
                min_order = min(slow_expected_order, fast_expected_order)
                max_order = max(slow_expected_order, fast_expected_order)
                atol = 0.3
                @test (
                    isapprox(rate[end], min_order; atol = atol) ||
                    isapprox(rate[end], max_order; atol = atol) ||
                    min_order <= rate[end] <= max_order
                )
            end
        end
    end
end
