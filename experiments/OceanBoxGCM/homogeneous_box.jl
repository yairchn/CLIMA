using Test
using CLIMA
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.VariableTemplates
using CLIMA.Mesh.Grids: polynomialorder
using CLIMA.DGmethods: vars_state
using CLIMA.HydrostaticBoussinesq

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
    if BC == nothing
        problem = HomogeneousBox{FT}(dimensions...)
    else
        problem = HomogeneousBox{FT}(dimensions...; BC = BC)
    end

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = cʰ)

    config =
        CLIMA.OceanBoxGCMConfiguration("homogeneous_box", N, resolution, model)

    return config
end

function run_homogeneous_box(; imex::Bool = false, BC = nothing)
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(20)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 4e6   # m
    Lʸ = 4e6   # m
    H = 400   # m
    dimensions = (Lˣ, Lʸ, H)

    timestart = FT(0)    # s
    timeend = FT(6 * 3600) # s

    if imex
        solver_type = CLIMA.IMEXSolverType(
            linear_model = LinearHBModel,
            linear_solver = CLIMA.ColumnwiseLUSolver.SingleColumnLU,
        )
        Nᶻ = Int(400)
        Courant_number = 0.1
    else
        solver_type =
            CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
        Courant_number = 0.4
    end

    driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver_config = CLIMA.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_solver_type = solver_type,
        modeldata = modeldata,
        Courant_number = Courant_number,
    )

    result = CLIMA.invoke!(solver_config)

    maxQ = Vars{vars_state(driver_config.bl, FT)}(maximum(
        solver_config.Q,
        dims = (1, 3),
    ))
    minQ = Vars{vars_state(driver_config.bl, FT)}(minimum(
        solver_config.Q,
        dims = (1, 3),
    ))

    @test maxQ.θ ≈ minQ.θ
end

@testset "$(@__FILE__)" begin
    boundary_conditions = [
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressNoForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressNoForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressNoForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressNoForcing(),
        ),
    ]

    for BC in boundary_conditions
        run_homogeneous_box(imex = true, BC = BC)
        run_homogeneous_box(imex = false, BC = BC)
    end
end
