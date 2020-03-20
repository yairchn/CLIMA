using Dates
using DocStringExtensions
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.ConfigTypes
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Grids
using CLIMA.GenericCallbacks
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Topologies
using CLIMA.MoistThermodynamics
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
using CLIMA.VTK

import CLIMA.DGmethods:
    BalanceLaw,
    DGModel,
    LocalGeometry,
    vars_state,
    vars_aux,
    vars_gradient,
    vars_diffusive,
    init_state!,
    init_aux!,
    update_aux!,
    nodal_update_aux!,
    flux_nondiffusive!,
    flux_diffusive!,
    wavespeed,
    boundary_state!,
    source!

# ------------------------ Description ------------------------- #
# The test is based on a modelling set-up designed for the
# 8th International Cloud Modelling Workshop
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See chapter 2 in Arabas et al 2015 for setup details:
#@Article{gmd-8-1677-2015,
#AUTHOR = {Arabas, S. and Jaruga, A. and Pawlowska, H. and Grabowski, W. W.},
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk,
#         and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}
# ------------------------ Description ------------------------- #

Base.@kwdef struct KinematicBC{M, E, Q}
    momentum::M = Impenetrable(FreeSlip())
    energy::E = Insulating()
    moisture::Q = Impermeable()
end

struct KinematicModelConfig{FT}
    xmax::FT
    ymax::FT
    zmax::FT
    wmax::FT
    θ_0::FT
    p_0::FT
    p_1000::FT
    qt_0::FT
    z_0::FT
end

struct KinematicModel{FT, O, M, P, S, BC, IS, DC} <: BalanceLaw
    orientation::O
    moisture::M
    precipitation::P
    source::S
    boundarycondition::BC
    init_state::IS
    data_config::DC
end

function KinematicModel{FT}(
    ::Type{AtmosLESConfigType};
    orientation::O = FlatOrientation(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    source::S = nothing,
    boundarycondition::BC = NoFluxBC(),
    init_state::IS = nothing,
    data_config::DC = nothing,
) where {FT <: AbstractFloat, O, M, P, S, BC, IS, DC}

    @assert init_state ≠ nothing

    atmos = (
        orientation,
        moisture,
        precipitation,
        source,
        boundarycondition,
        init_state,
        data_config,
    )

    return KinematicModel{FT, typeof.(atmos)...}(atmos...)
end

function vars_state(m::KinematicModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
    end
end

function vars_aux(m::KinematicModel, FT)
    @vars begin
        # defined in init_aux
        p::FT
        x::FT
        y::FT
        z::FT
        # defined in update_aux
        u::FT
        w::FT
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        T::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
    end
end

vars_gradient(m::KinematicModel, FT) = @vars()

vars_diffusive(m::KinematicModel, FT) = @vars()

function init_aux!(m::KinematicModel, aux::Vars, geom::LocalGeometry)

    FT = eltype(aux)
    x, y, z = geom.coord
    dc = m.data_config

    # TODO - should R_d and cp_d here be R_m and cp_m?
    R_m, cp_m, cv_m, γ = gas_constants(PhasePartition(dc.qt_0))

    # Pressure profile assuming hydrostatic and constant θ and qt profiles.
    # It is done this way to be consistent with Arabas paper.
    # It's not neccesarily the best way to initialize with our model variables.
    p =
        dc.p_1000 *
        (
            (dc.p_0 / dc.p_1000)^(R_d / cp_d) -
            R_d / cp_d * grav / dc.θ_0 / R_m * (z - dc.z_0)
        )^(cp_d / R_d)
    aux.p = p
    aux.x = x
    aux.y = y
    aux.z = z
end

function init_state!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    m.init_state(m, state, aux, coords, t, args...)
end

function init_kinematic_eddy!(eddy_model, state, aux, (x, y, z), t)
    FT = eltype(state)
    dc = eddy_model.data_config

    # density
    q_pt_0 = PhasePartition(dc.qt_0)
    R_m, cp_m, cv_m, γ = gas_constants(q_pt_0)
    T::FT = dc.θ_0 * (aux.p / dc.p_1000)^(R_m / cp_m)
    ρ::FT = aux.p / R_m / T
    state.ρ = ρ

    # moisture
    state.ρq_tot = ρ * dc.qt_0

    # velocity (derivative of streamfunction)
    ρu::FT =
        dc.wmax * dc.xmax / dc.zmax *
        cos(π * z / dc.zmax) *
        cos(2 * π * x / dc.xmax)
    ρw::FT = 2 * dc.wmax * sin(π * z / dc.zmax) * sin(2 * π * x / dc.xmax)
    state.ρu = SVector(ρu, FT(0), ρw)
    u::FT = ρu / ρ
    w::FT = ρw / ρ

    # energy
    e_kin::FT = 1 // 2 * (u^2 + w^2)
    e_pot::FT = grav * z
    e_int::FT = internal_energy(T, q_pt_0)
    e_tot::FT = e_kin + e_pot + e_int
    state.ρe = ρ * e_tot

    return nothing
end

function update_aux!(dg::DGModel, m::KinematicModel, Q::MPIStateArray, t::Real)
    nodal_update_aux!(kinematic_model_nodal_update_aux!, dg, m, Q, t)

    return true
end

function kinematic_model_nodal_update_aux!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)

    aux.u = state.ρu[1] / state.ρ
    aux.w = state.ρu[3] / state.ρ

    aux.q_tot = state.ρq_tot / state.ρ

    aux.e_tot = state.ρe / state.ρ
    aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
    aux.e_pot = grav * aux.z
    aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot

    # saturation adjustment happens here
    ts = PhaseEquil(aux.e_int, state.ρ, aux.q_tot)
    pp = PhasePartition(ts)

    aux.T = ts.T

    aux.q_vap = aux.q_tot - pp.liq - pp.ice
    aux.q_liq = pp.liq
    aux.q_ice = pp.ice

end

function boundary_state!(nf, m::KinematicModel, args...)
    bc_kin_boundary_state!(nf, m.boundarycondition, m, args...)
end
@generated function bc_kin_boundary_state!(
    nf,
    tup::Tuple,
    m,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(tup)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> bc_kin_boundary_state!(
                nf,
                tup[i],
                m,
                state⁺,
                aux⁺,
                n,
                state⁻,
                aux⁻,
                bctype,
                t,
                args...,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

function bc_kin_boundary_state!(nf, bc::KinematicBC, m, args...)
    # bc_kin_momentum_boundary_state!(nf, bc.momentum, m, args...)
    # bc_kin_energy_boundary_state!(nf, bc.energy, m, args...)
    # bc_kin_moisture_boundary_state!(nf, bc.moisture, m, args...)
end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    return abs(dot(nM, u))
end

@inline function flux_nondiffusive!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    FT = eltype(state)

    # advect moisture ...
    flux.ρq_tot =
        SVector(state.ρu[1] * aux.q_tot, FT(0), state.ρu[3] * aux.q_tot)
    # ... energy ...
    flux.ρe =
        SVector(aux.u * (state.ρe + aux.p), FT(0), aux.w * (state.ρe + aux.p))
    # ... and don't advect momentum (kinematic setup)
end

@inline function flux_diffusive!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) end

@inline function flux_diffusive!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    τ,
    d_h_tot,
) end

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) end

function config_kinematic_eddy(
    FT,
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    wmax,
    θ_0,
    p_0,
    p_1000,
    qt_0,
    z_0,
)
    # Choose explicit solver
    ode_solver =
        CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)

    kmc = KinematicModelConfig(
        FT(xmax),
        FT(ymax),
        FT(zmax),
        FT(wmax),
        FT(θ_0),
        FT(p_0),
        FT(p_1000),
        FT(qt_0),
        FT(z_0),
    )

    # Set up the model
    model = KinematicModel{FT}(
        AtmosLESConfigType;
        boundarycondition = KinematicBC(),
        init_state = init_kinematic_eddy!,
        data_config = kmc,
    )

    config = CLIMA.AtmosLESConfiguration(
        "KinematicModel",
        N,
        resolution,
        FT(xmax),
        FT(ymax),
        FT(zmax),
        init_kinematic_eddy!;
        solver_type = ode_solver,
        model = model,
    )

    return config
end

function main()
    CLIMA.init()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(20)
    Δy = FT(1)
    Δz = FT(20)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 1500
    ymax = 10
    zmax = 1500
    # initial configuration
    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]
    z_0 = FT(0) # surface height

    # Simulation time
    t0 = FT(0)
    # timeend = FT(30 * 60)
    timeend = FT(60 * 30)
    # Courant number
    CFL = FT(0.8)

    driver_config = config_kinematic_eddy(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        wmax,
        θ_0,
        p_0,
        p_1000,
        qt_0,
        z_0,
    )
    solver_config = CLIMA.setup_solver(
        t0,
        timeend,
        driver_config;
        ode_dt = FT(1),
        init_on_cpu = true,
        Courant_number = CFL,
    )

    mpicomm = MPI.COMM_WORLD

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(solver_config.Q)
            @info @sprintf(
                """Update
                norm(Q) = %.16e""",
                energy
            )
        end
    end

    # output for paraview
    model = driver_config.bl
    step = [0]
    cbvtk = GenericCallbacks.EveryXSimulationSteps(60) do (init = false)
        mkpath("vtk/")
        outprefix = @sprintf(
            "vtk/new_ex_1_mpirank%04d_step%04d",
            MPI.Comm_rank(mpicomm),
            step[1]
        )
        @info "doing VTK output" outprefix
        writevtk(
            outprefix,
            solver_config.Q,
            solver_config.dg,
            flattenednames(vars_state(model, FT)),
            solver_config.dg.auxstate,
            flattenednames(vars_aux(model, FT)),
        )
        step[1] += 1
        nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = CLIMA.invoke!(
        solver_config;
        user_callbacks = (cbtmarfilter, cbinfo, cbvtk),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
