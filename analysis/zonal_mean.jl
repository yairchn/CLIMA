using NCDatasets
using Plots
using Statistics
using DataFrames
using Printf


# analysis folder
ana_folder = "/central/scratch/bischtob/heldsuarez/nc"
num_avg = 6

# read the Netcdf files from disk
function read_ncfiles(filename, pinfo)
  path = joinpath(ana_folder, filename)
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

# get variables
lon, lat, rad, ρ, u, e = read_ncfiles("hs_test_step0001.nc", true)

# load data into arrays for averaging
ρ_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
u_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
e_a = zeros(num_avg, size(rad)[1], size(lat)[1], size(lon)[1])
for i in 1:num_avg
  lon, lat, rad, ρ_a[i,:,:,:], u_a[i,:,:,:], e_a[i,:,:,:] = read_ncfiles(@sprintf("hs_test_step000%s.nc",i), false)
end

# average data
u_zm = dropdims( mean(u_a, dims = (1,4) ) , dims = (1,4) )
e_zm = dropdims( mean(e_a, dims = (1,4) ) , dims = (1,4) )

# plot
pyplot()
contourf(lat*180/3.14 .- 90.0, rad, u_zm, levels=10)

xlabel!("latitude")
ylabel!("z [m]")
savefig("zonal_mean.png")

