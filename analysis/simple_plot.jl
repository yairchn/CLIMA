#import Pkg; Pkg.add("DataFrames")


using NCDatasets
using Plots
using Statistics
using DataFrames
using Printf

function read_ncfiles(filename, pinfo)
  path = joinpath("../output/nc/",filename)
  dfile = Dataset(path)
  if pinfo==true
    print(dfile.attrib)
  end
  ρ = dfile["ro"].var[:]
  u = dfile["rou"].var[:] ./ ρ
  e = dfile["roe"].var[:] ./ ρ
  lat = dfile["lat"].var[:]
  lon = dfile["long"].var[:]
  rad = dfile["rad"].var[:] .- 6371000.0
  return lon, lat, rad, ρ, u, e
end


# time indep
lon, lat, rad, ρ, u, e = read_ncfiles("hs_test_step0001.nc", true)

# load data
ρ_a = zeros(6, size(rad)[1], size(lat)[1], size(lon)[1])
u_a = zeros(6, size(rad)[1], size(lat)[1], size(lon)[1])
e_a = zeros(6, size(rad)[1], size(lat)[1], size(lon)[1])
for i in 1:6
  lon, lat, rad, ρ_a[i,:,:,:], u_a[i,:,:,:], e_a[i,:,:,:] = read_ncfiles(@sprintf("hs_test_step000%s.nc",i), false)
end

# average data
u_zm = dropdims( mean(u_a, dims = (1,4) ) , dims = (1,4) )
e_zm = dropdims( mean(e_a, dims = (1,4) ) , dims = (1,4) )

# plot
plot(contourf(lat*180/3.14,rad,u_zm,levels=10), contourf(lat*180/3.14,rad,e_zm,levels=10))

xlabel!("lat")
ylabel!("z")
title!("zonal mean u plot")
plot!(layout=1)
png("the_plot")



