function update3N!(dQ, Q1, Q2, Q3, γ1, γ2, γ3, β, δ, dt, first_stage)
  @inbounds @loop for i = (1:length(Q1);
                           (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    q1 = Q1[i]
    # If this is the first stage initialize the scratch arrays
    if first_stage
      q3 = Q3[i] = q1
      q2 = zero(eltype(q1))
    else
      q2, q3 = Q2[i], Q3[i]
    end

    # Update the stratch array
    Q2[i] = q2 = q2 + δ * q1

    # update the solution
    Q1[i] = γ1 * q1 + γ2 * q2 + γ3 * q3 + β * dt * dQ[i]
  end
end
