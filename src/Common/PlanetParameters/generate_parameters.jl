
function generate_parameters(filename, planet)

open(filename, "w") do io
contents = "module Parameters

# Only exporting grav for now
export grav

export ParameterSet
export parameter_set

struct ParameterSet{P} end

const parameter_set = ParameterSet{$planet}

# Physical constants
gas_constant(ps::Type{ParameterSet})   = 8.3144598                            # Universal gas constant (J/mol/K)
light_speed(ps::Type{ParameterSet})    = 2.99792458e8                         # Speed of light in vacuum (m/s)
h_Planck(ps::Type{ParameterSet})       = 6.626e-34                            # Planck constant (m^2 kg/s)
k_Boltzmann(ps::Type{ParameterSet})    = 1.381e-23                            # Boltzmann constant (m^2 kg/s^2/K)
Stefan(ps::Type{ParameterSet})         = 5.670e-8                             # Stefan-Boltzmann constant (W/m^2/K^4)
astro_unit(ps::Type{ParameterSet})     = 1.4959787e11                         # Astronomical unit (m)
k_Karman(ps::Type{ParameterSet})       = 0.4                                  # Von Karman constant (1)

# Properties of dry air
molmass_dryair(ps::Type{ParameterSet}) = 28.97e-3                             # Molecular weight dry air (kg/mol)
R_d(ps::Type{ParameterSet})            = gas_constant(ps)/molmass_dryair(ps)  # Gas constant dry air (J/kg/K)
kappa_d(ps::Type{ParameterSet})        = 2//7                                 # Adiabatic exponent dry air
cp_d(ps::Type{ParameterSet})           = R_d(ps)/kappa_d(ps)                  # Isobaric specific heat dry air
cv_d(ps::Type{ParameterSet})           = cp_d(ps) - R_d(ps)                   # Isochoric specific heat dry air

# Properties of water
ρ_cloud_liq(ps::Type{ParameterSet})    = 1e3                                  # Density of liquid water (kg/m^3)
ρ_cloud_ice(ps::Type{ParameterSet})    = 916.7                                # Density of ice water (kg/m^3)
molmass_water(ps::Type{ParameterSet})  = 18.01528e-3                          # Molecular weight (kg/mol)
molmass_ratio(ps::Type{ParameterSet})  = molmass_dryair(ps)/molmass_water(ps) # Molar mass ratio dry air/water
R_v(ps::Type{ParameterSet})            = gas_constant(ps)/molmass_water(ps)   # Gas constant water vapor (J/kg/K)
cp_v(ps::Type{ParameterSet})           = 1859                                 # Isobaric specific heat vapor (J/kg/K)
cp_l(ps::Type{ParameterSet})           = 4181                                 # Isobaric specific heat liquid (J/kg/K)
cp_i(ps::Type{ParameterSet})           = 2100                                 # Isobaric specific heat ice (J/kg/K)
cv_v(ps::Type{ParameterSet})           = cp_v(ps) - R_v(ps)                   # Isochoric specific heat vapor (J/kg/K)
cv_l(ps::Type{ParameterSet})           = cp_l(ps)                             # Isochoric specific heat liquid (J/kg/K)
cv_i(ps::Type{ParameterSet})           = cp_i(ps)                             # Isochoric specific heat ice (J/kg/K)
T_freeze(ps::Type{ParameterSet})       = 273.15                               # Freezing point temperature (K)
T_min(ps::Type{ParameterSet})          = 150.0                                # Minimum temperature guess in saturation adjustment (K)
T_max(ps::Type{ParameterSet})          = 1000.0                               # Maximum temperature guess in saturation adjustment (K)
T_icenuc(ps::Type{ParameterSet})       = 233.00                               # Homogeneous nucleation temperature (K)
T_triple(ps::Type{ParameterSet})       = 273.16                               # Triple point temperature (K)
T_0(ps::Type{ParameterSet})            = T_triple(ps)                         # Reference temperature (K)
LH_v0(ps::Type{ParameterSet})          = 2.5008e6                             # Latent heat vaporization at T_0 (J/kg)
LH_s0(ps::Type{ParameterSet})          = 2.8344e6                             # Latent heat sublimation at T_0 (J/kg)
LH_f0(ps::Type{ParameterSet})          = LH_s0(ps) - LH_v0(ps)                # Latent heat of fusion at T_0 (J/kg)
e_int_v0(ps::Type{ParameterSet})       = LH_v0(ps) - R_v(ps)*T_0(ps)          # Specific internal energy of vapor at T_0 (J/kg)
e_int_i0(ps::Type{ParameterSet})       = LH_f0(ps)                            # Specific internal energy of ice at T_0 (J/kg)
press_triple(ps::Type{ParameterSet})   = 611.657                              # Triple point vapor pressure (Pa)

# Properties of sea water
ρ_ocean(ps::Type{ParameterSet})        = 1.035e3                              # Reference density sea water (kg/m^3)
cp_ocean(ps::Type{ParameterSet})       = 3989.25                              # Specific heat sea water (J/kg/K)

# Planetary parameters
planet_radius(ps::Type{ParameterSet})  = 6.371e6                              # Mean planetary radius (m)
day(ps::Type{ParameterSet})            = 86400                                # Length of day (s)
Omega(ps::Type{ParameterSet})          = 7.2921159e-5                         # Ang. velocity planetary rotation (1/s)
grav(ps::Type{ParameterSet})           = 9.81                                 # Gravitational acceleration (m/s^2)
year_anom(ps::Type{ParameterSet})      = 365.26*day(ps)                       # Length of anomalistic year (s)
orbit_semimaj(ps::Type{ParameterSet})  = 1*astro_unit(ps)                     # Length of semimajor orbital axis (m)
TSI(ps::Type{ParameterSet})            = 1362                                 # Total solar irradiance (W/m^2)
MSLP(ps::Type{ParameterSet})           = 1.01325e5                            # Mean sea level pressure (Pa)

end"
  print(io, contents)

end

end


file = "Parameters.jl"
generate_parameters(file, ":Earth")

include(file)

using Main.Parameters

@show grav(parameter_set)


