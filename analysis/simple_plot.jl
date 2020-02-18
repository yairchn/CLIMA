#import Pkg; Pkg.add("DataFrames")


using NCDatasets
using Plots
using Statistics
using DataFrames
using Printf

function read_ncfiles(filename)
  path = joinpath("../output/nc/",filename)
  dfile = Dataset(path)
  ρ = dfile["ro"].var[:]
  u = dfile["rou"].var[:] ./ ρ
  lat = dfile["lat"].var[:]
  lon = dfile["long"].var[:]
  rad = dfile["rad"].var[:] .- 6371000.0
  return lon, lat, rad, ρ, u
end


# time indep
lon, lat, rad, ρ, u = read_ncfiles("hs_test_step0001_notworking.nc")


ρ_a = zeros(6, size(rad)[1], size(lat)[1], size(lon)[1])
u_a = zeros(6, size(rad)[1], size(lat)[1], size(lon)[1])  
for i in 1:6
  lon, lat, rad, ρ_a[i,:,:,:], u_a[i,:,:,:] = read_ncfiles(@sprintf("hs_test_step000%s_notworking.nc",i))
end

# load data

u_zm = dropdims( mean(u_a, dims = (1,4) ) , dims = (1,4) )

# plot
plot(lat*180/3.14,rad,u_zm)

xlabel!("lat")
ylabel!("z")
title!("zonal mean u plot")

png("the_plot")



