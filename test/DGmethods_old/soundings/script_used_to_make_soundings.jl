using DelimitedFiles
using NCDatasets

D = Dict()
D["SOUNDING_PYCLES_Z_T_P"] = 0
D["sounding_DYCOMS_TEST1"] = 0
D["sounding_JCP2013"] = 0
D["sounding_WKR88"] = 0
D["sounding_GC1991"] = 0
D["sounding_JCP2013_with_pressure"] = 0
D["sounding_blend"] = 0

D_nvars = Dict()
D_nvars["SOUNDING_PYCLES_Z_T_P"] = 5
D_nvars["sounding_DYCOMS_TEST1"] = 6
D_nvars["sounding_JCP2013"] = 5
D_nvars["sounding_WKR88"] = 5
D_nvars["sounding_GC1991"] = 6
D_nvars["sounding_JCP2013_with_pressure"] = 6
D_nvars["sounding_blend"] = 5

FT = Float64
for k in keys(D)
  D[k] = Array{FT}(readdlm("$k.dat", ','))
end

vars = ("height [m]", "theta [K]", "qv [g kg⁻¹]", "u [m s⁻¹]", "v [m s⁻¹]", "pressure [Pa]")

for k in keys(D)
  ds = Dataset("$k.nc","c")
  for (i,v) in enumerate(vars[1:D_nvars[k]])
    defVar(ds, v, D[k][:,i], ("datapoint",))
  end
  close(ds)
end
