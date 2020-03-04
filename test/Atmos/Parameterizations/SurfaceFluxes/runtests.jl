using Test

using CLIMA.SurfaceFluxes
using CLIMA.MoistThermodynamics
using CLIMA.RootSolvers

# FIXME: Use realistic values / test for correctness
# These tests have been run to ensure they do not fail,
# but they need further testing for correctness.

@testset "SurfaceFluxes" begin

  x_initial = [100, 15.0, 350.0]
  F_exchange = [2.0, 3.0]
  z_0 = [1.0, 1.0]
  dimensionless_number = [1.0, 0.74]
  x_ave = [5.0, 350.0]
  x_s   = [0.0, 300.0]

  Δz = 2.0
  z = 0.5
  θ_bar = 300.0
  a = 4.7
  pottemp_flux_given = -200.0
  args = x_initial, x_ave, x_s, z_0, F_exchange, dimensionless_number, θ_bar, Δz, z, a, pottemp_flux_given
  sfc = surface_conditions(args[1:end-1]...)

  @test sfc.L_MO                   ≈ 54.563405359719404
  @test sfc.pottemp_flux_star      ≈ -1132.9097989525164
  @test all(sfc.flux              .≈ [-86.79000158448329, -1132.9097989138224])
  @test all(sfc.x_star            .≈ [9.316115155175106, 121.60753490520031])
  @test all(sfc.K_exchange        .≈ [0.9969164175880834, 0.08477271454000379])

  sfc = surface_conditions(args...)

  @test sfc.L_MO                   ≈ 405.8862509767147
  @test sfc.pottemp_flux_star      ≈ -199.99999999999997
  @test all(sfc.flux              .≈ [-104.07858678549196, -1399.2078659154145])
  @test all(sfc.x_star            .≈ [10.20189133374258, 137.15181039887756])
  @test all(sfc.K_exchange        .≈ [6.771981702484999, 0.5591365770421184])

end

