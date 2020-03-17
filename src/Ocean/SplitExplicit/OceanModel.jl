struct OceanModel{P,T} <: AbstractOceanModel
  problem::P
  cʰ::T
  cᶻ::T
  αᵀ::T
  νʰ::T
  νᶻ::T
  κʰ::T
  κᶻ::T
  function OceanModel{FT}(problem;
                          cʰ = FT(0),     # m/s
                          cᶻ = FT(0),     # m/s
                          αᵀ = FT(2e-4),  # (m/s)^2 / K
                          νʰ = FT(5e3),   # m^2 / s
                          νᶻ = FT(5e-3),  # m^2 / s
                          κʰ = FT(1e3),   # m^2 / s
                          κᶻ = FT(1e-4),  # m^2 / s
                          ) where {FT <: AbstractFloat}
    return new{typeof(problem),FT}(problem, cʰ, cᶻ, αᵀ, νʰ, νᶻ, κʰ, κᶻ)
  end
end

function calculate_dt(grid, model::OceanModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())
    minΔz = min_node_distance(grid, VerticalDirection())

    CFL_gravity = minΔx / model.cʰ
    CFL_diffusive = minΔz^2 / (1000 * model.κᶻ)
    CFL_viscous = minΔz^2 / model.νᶻ

    dt = 1//2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

"""
    OceanDGModel()

helper function to add required filtering
not used in the Driver+Config setup
"""
function OceanDGModel(bl::OceanModel, grid, numfluxnondiff, numfluxdiff,
                      gradnumflux; kwargs...)
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)

  tendency_dg = DGModel(TendencyIntegralModel(bl), grid, numfluxnondiff, numfluxdiff, gradnumflux)

  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter, tendency_dg = tendency_dg)

  return DGModel(bl, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 kwargs..., modeldata=modeldata)
end

function vars_state(m::OceanModel, T)
    @vars begin
        u::SVector{2, T}
        η::T
        θ::T
    end
end

function init_state!(m::OceanModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

function vars_aux(m::OceanModel, T)
    @vars begin
        w::T
        pkin_reverse::T # ∫(-αᵀ θ)
        w_reverse::T
        pkin::T         # ∫(-αᵀ θ)
        wz0::T          # w at z=0
        ν::SVector{3, T}
        κ::SVector{3, T}
        f::T            # coriolis
        τ::T            # wind stress  # TODO: Should be 2D
        θʳ::T           # SST given    # TODO: Should be 2D
        ∫u::T
    end
end

function init_aux!(m::OceanModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

function vars_gradient(m::OceanModel, T)
  @vars begin
    u::SVector{2, T}
    θ::T
  end
end

@inline function gradvariables!(m::OceanModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

function vars_diffusive(m::OceanModel, T)
  @vars begin
    ∇u::SMatrix{3, 2, T, 6}
    ∇θ::SVector{3, T}
  end
end

@inline function diffusive!(m::OceanModel, D::Vars, G::Grad,
                            Q::Vars, A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

function vars_integrals(m::OceanModel, T)
  @vars begin
    ∇hu::T
    αᵀθ::T
  end
end

@inline function integrate_aux!(m::OceanModel, integrand::Vars, Q::Vars, A::Vars)
  αᵀ = m.αᵀ
  integrand.αᵀθ = -αᵀ * Q.θ
  integrand.∇hu = A.w # borrow the w value from A...

  return nothing
end

@inline function flux_nondiffusive!(m::OceanModel, F::Grad, Q::Vars,
                                    A::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    θ = Q.θ
    w = A.w   # vertical velocity
    pkin = A.pkin
    v = @SVector [u[1], u[2], w]

    # ∇ • (u θ)
    F.θ += v * θ

    # ∇h • (- ∫(αᵀ θ))
    F.u += grav * pkin * Ih

    # ∇h • (v ⊗ u)
    # F.u += v * u'
  end

  return nothing
end

@inline function flux_diffusive!(m::OceanModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
    # horizontal viscosity done in horizontal model
    F.u -= Diagonal([-0, -0, A.ν[3]]) * D.∇u
    F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

@inline function source!(m::OceanModel{P}, S::Vars, Q::Vars, A::Vars,
                         t::Real) where P
    @inbounds begin
        u = Q.u # Horizontal components of velocity
        f = A.f

        # f × u
        S.u -= @SVector [-f * u[2], f * u[1]]

        S.η += A.wz0
    end

    return nothing
end

function update_aux!(dg::DGModel, m::OceanModel, Q::MPIStateArray, t::Real)
  MD = dg.modeldata

  # required to ensure that after integration velocity field is divergence free
  vert_filter = MD.vert_filter
  # Q[1] = u[1] = u, Q[2] = u[2] = v
  apply!(Q, (1, 2), dg.grid, vert_filter, VerticalDirection())

  exp_filter = MD.exp_filter
  # Q[4] = θ
  apply!(Q, (4,), dg.grid, exp_filter, VerticalDirection())

  return true
end

function update_aux_diffusive!(dg::DGModel, m::OceanModel, Q::MPIStateArray, t::Real)
    A  = dg.auxstate

    # store ∇ʰu as integrand for w
    # update vertical diffusivity for convective adjustment
    function f!(::OceanModel, Q, A, D, t)
        @inbounds begin
            A.w = -(D.∇u[1,1] + D.∇u[2,2])

            D.∇θ[3] < 0 ? A.κ = (m.κʰ, m.κʰ, 1000 * m.κᶻ) : A.κ = (m.κʰ, m.κʰ, m.κᶻ)
        end

        return nothing
    end
    nodal_update_aux!(f!, dg, m, Q, t; diffusive=true)

    # compute integrals for w and pkin
    indefinite_stack_integral!(dg, m, Q, A, t) # bottom -> top
    reverse_indefinite_stack_integral!(dg, m, A, t) # top -> bottom

    # copy down wz0
    copy_stack_field_down!(dg, m, A, 5, 1)

    return true
end

@inline wavespeed(m::OceanModel, n⁻, _...) = abs(SVector(m.cʰ, m.cʰ, m.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(::Rusanov, ::OceanModel, n⁻, λ, ΔQ::Vars,
                         Q⁻, A⁻, Q⁺, A⁺, t)
    ΔQ.η = -0

    return nothing
end
