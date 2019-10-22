#### InitParams

export Params

"""
    Params(::Case)

Initialize stand-alone input parameters to
solve the EDMF equations for a given case.
"""
function Params end

function Params(::BOMEX)
  FT = Float64
  params = Dict()
  params[:export_data] = false
  params[:plot_single_fields] = true
  params[:export_frequency] = 2000

  params[:ForcingType] = StandardForcing(apply_subsidence=true,
                                         apply_coriolis=true,
                                         coriolis_param=FT(0.376e-4))
  params[:EntrDetrModel] = BOverW2{FT}(1.0, 1.0)
  params[:MixingLengthModel] = ConstantMixingLength{FT}(100)
  # params[:MixingLengthModel] = SCAMPyMixingLength{FT}(StabilityDependentParam{FT}(2.7,-100.0),
  #                                                     StabilityDependentParam{FT}(-1.0,-0.2))

  # params[:MixingLengthModel] = IgnaciosMixingLength(StabilityDependentParam{FT}(2.7,-100.0),
  #                                                   StabilityDependentParam{FT}(-1.0,-0.2),
  #                                                   0.1, 0.12, 0.4, 40/13)
  params[:EddyDiffusivityModel]           = SCAMPyEddyDiffusivity{FT}(0.1)
  params[:PressureModel]    = SCAMPyPressure{FT}(1.0/3.0, 0.375, 500.0)

  params[:GridParams] = GridParams(z_min=FT(0.0),
                                   z_max=FT(3000.0),
                                   n_elems=75)

  params[:TimeMarchingParams] = TimeMarchingParams(Δt=FT(20.0),
                                                   Δt_min=FT(20.0),
                                                   t_end=FT(21600.0),
                                                   CFL=FT(0.8),
                                                   )
  params[:Δt] = FT(20.0)
  params[:Δt_min] = FT(20.0)
  params[:t_end] = FT(21600.0)
  params[:CFL] = FT(0.8)

  params[:N_subdomains] = 3
  # TOFIX: Remove indexes from Params

  params[:prandtl_number]         = FT(1.0)
  params[:tke_diss_coeff]         = FT(2.0)


  params[:a_bounds] = [1e-3, 1-1e-3]                                  # filter for a
  params[:w_bounds] = [0.0, 10000.0]                                  # filter for w
  params[:q_bounds] = [0.0, 1.0]                                      # filter for q

  params[:Prandtl_neutral] = 0.74

  params[:SurfaceType] = SurfaceFixedFlux(T=FT(300.4),
                                          P=FT(1.015e5),
                                          q_tot=FT(0.02245),
                                          ustar=FT(0.28),
                                          windspeed_min=FT(0.0),
                                          tke_tol=FT(0.01),
                                          area=FT(0.1)
                                          )


  params[:Ri_bulk_crit] = 0.0                                         # inversion height parameters
  params[:bflux] = (grav * ((8.0e-3 + (molmass_ratio-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (molmass_ratio-1) * 22.45e-3))))

  return params
end
