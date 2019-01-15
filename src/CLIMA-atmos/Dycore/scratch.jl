using LinearAlgebra

flux(a, b) = (a + b)^2 / 2

function naiveT!(DrF, DsF, DtF, D, Q)
  F = flux.(Q, Q')
  eye = Matrix{Float64}(I, size(D))
  Dr = kron(eye, eye, D)
  Ds = kron(eye, D, eye)
  Dt = kron(D, eye, eye)

  sum!(DrF, Dr' .* F)
  sum!(DsF, Ds' .* F)
  sum!(DtF, Dt' .* F)
end

function naive!(DrF, DsF, DtF, D, Q)
  F = flux.(Q, Q')
  eye = Matrix{Float64}(I, size(D))
  Dr = kron(eye, eye, D)
  Ds = kron(eye, D, eye)
  Dt = kron(D, eye, eye)

  sum!(DrF, Dr .* F)
  sum!(DsF, Ds .* F)
  sum!(DtF, Dt .* F)
end

function lineT!(DrF, DsF, DtF, D, Q)
  Nq = size(D, 1)
  Q = reshape(Q, Nq, Nq, Nq)

  @inbounds for k = 1:Nq
    for j = 1:Nq
      for i = 1:Nq
        Fr = 0
        Fs = 0
        Ft = 0
        for n = 1:Nq
          Fr += D[n,i] * flux(Q[n,j,k], Q[i,j,k])
          Fs += D[n,j] * flux(Q[i,n,k], Q[i,j,k])
          Ft += D[n,k] * flux(Q[i,j,n], Q[i,j,k])
        end
        DrF[i + Nq * (j-1) + Nq^2 * (k-1)] = Fr
        DsF[i + Nq * (j-1) + Nq^2 * (k-1)] = Fs
        DtF[i + Nq * (j-1) + Nq^2 * (k-1)] = Ft
      end
    end
  end
end

function line!(DrF, DsF, DtF, D, Q)
  Nq = size(D, 1)
  Q = reshape(Q, Nq, Nq, Nq)

  @inbounds for k = 1:Nq
    for j = 1:Nq
      for i = 1:Nq
        Fr = 0
        Fs = 0
        Ft = 0
        for n = 1:Nq
          Fr += D[i,n] * flux(Q[n,j,k], Q[i,j,k])
          Fs += D[j,n] * flux(Q[i,n,k], Q[i,j,k])
          Ft += D[k,n] * flux(Q[i,j,n], Q[i,j,k])
        end
        DrF[i + Nq * (j-1) + Nq^2 * (k-1)] = Fr
        DsF[i + Nq * (j-1) + Nq^2 * (k-1)] = Fs
        DtF[i + Nq * (j-1) + Nq^2 * (k-1)] = Ft
      end
    end
  end
end


let
  N = 3
  Nq = N+1

  D = rand(Nq, Nq)
  Q = rand(Nq^3)

  out_naive = (similar(Q), similar(Q), similar(Q))
  naive!(out_naive..., D, Q)

  out_line = (similar(Q), similar(Q), similar(Q))
  line!(out_line..., D, Q)

  @assert out_naive[1] ≈ out_line[1]
  @assert out_naive[2] ≈ out_line[2]
  @assert out_naive[3] ≈ out_line[3]

  nothing
end

let
  N = 3
  Nq = N+1

  D = rand(Nq, Nq)
  Q = rand(Nq^3)

  out_naive = (similar(Q), similar(Q), similar(Q))
  naiveT!(out_naive..., D, Q)

  out_line = (similar(Q), similar(Q), similar(Q))
  lineT!(out_line..., D, Q)

  @assert out_naive[1] ≈ out_line[1]
  @assert out_naive[2] ≈ out_line[2]
  @assert out_naive[3] ≈ out_line[3]

  nothing
end
