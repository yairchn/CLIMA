using KernelAbstractions.Extras: @unroll

@kernel function stage_update!(
    Q,
    Qstages,
    Rstages,
    RKA,
    dt,
    ::Val{is},
    slow_δ,
    slow_dQ,
   ) where {is}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end
        Qstages_is_i = Q[i]
        @unroll for js in 1:(is - 1)
            Qstages_is_i += dt * RKA[is, js] * Rstages[js][i]
        end
        Qstages[is][i] = Qstages_is_i
    end
end

@kernel function solution_update!(
    Qnp1,
    Qn,
    error_estimate,
    Qstages,
    Rstages,
    RKB,
    RKB_embedded,
    dt,
    ::Val{Nstages},
    slow_δ,
    slow_dQ,
    slow_scaling,
   ) where {Nstages}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        Qnp1[i] = Qstages[4][i]
        error_estimate[i] = Qn[i] - Qnp1[i]
        @unroll for is in 1:Nstages
          error_estimate[i] += RKB_embedded[is] * dt * Rstages[is][i]
        end
    end
end
