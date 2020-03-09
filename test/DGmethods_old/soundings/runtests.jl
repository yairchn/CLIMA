using DataDeps
using NCDatasets
using Test

include("soundings_data.jl")

@testset "Get soundings data" begin
    data_folder = soundings_data_folder()
    PYCLES_Z_T_P          = Dataset(joinpath(data_folder, "SOUNDING_PYCLES_Z_T_P.nc"), "r")
    GC1991                = Dataset(joinpath(data_folder, "sounding_GC1991.nc"), "r")
    WKR88                 = Dataset(joinpath(data_folder, "sounding_WKR88.nc"), "r")
    JCP2013               = Dataset(joinpath(data_folder, "sounding_JCP2013.nc"), "r")
    blend                 = Dataset(joinpath(data_folder, "sounding_blend.nc"), "r")
    DYCOMS_TEST1          = Dataset(joinpath(data_folder, "sounding_DYCOMS_TEST1.nc"), "r")
    JCP2013_with_pressure = Dataset(joinpath(data_folder, "sounding_JCP2013_with_pressure.nc"), "r")
end
