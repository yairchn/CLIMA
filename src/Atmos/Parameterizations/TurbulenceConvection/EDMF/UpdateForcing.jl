#### UpdateForcing

function update_forcing! end

function update_forcing!(tmp::StateVec, q::StateVec, grid::Grid, params, ::Bomex)
  gm, en, ud, sd, al = allcombinations(q)
  for k in over_elems_real(grid)
    # Apply large-scale horizontal advection tendencies
    q_tot = q['q_tot', k, gm]
    q_vap = q_tot - tmp['q_liq', k, gm]
    q_tendencies['θ_liq', k, gm] += convert_forcing_thetal(tmp['p_0_half'][k],
                                                            q_tot,
                                                            q_vap,
                                                            tmp['T', k, gm],
                                                            tmp[:dqtdt, k],
                                                            tmp[:dTdt, k])
    q_tendencies['q_tot', k, gm] += self.dqtdt[k]
  end
  if params[:apply_subsidence]
      for k in grid.over_elems_real(Center()):
          # Apply large-scale subsidence tendencies
          q_tendencies['θ_liq', k, gm] -= grad(q['θ_liq', Dual(k), gm], grid) * tmp[:subsidence, k]
          q_tendencies['q_tot', k, gm] -= grad(q['q_tot', Dual(k), gm], grid) * tmp[:subsidence, k]

  if params[:apply_coriolis]
    coriolis_force!(grid, q, q_tendencies)
  end
end

