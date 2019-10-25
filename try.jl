using LinearAlgebra
α = rand()
β = rand()
γ = rand()
n = rand(3)
n = n / norm(n)

A1 = [0 1 0 0 0;
      α 0 0 0 β;
      0 0 0 0 0;
      0 0 0 0 0;
      0 γ 0 0 0]
A2 = [0 0 1 0 0;
      0 0 0 0 0;
      α 0 0 0 β;
      0 0 0 0 0;
      0 0 γ 0 0]
A3 = [0 0 0 1 0;
      0 0 0 0 0;
      0 0 0 0 0;
      α 0 0 0 β;
      0 0 0 γ 0]

An = n[1] * A1 + n[2] * A2 + n[3] * A3

Tn = [1 0    0     0   0;
      0 n[1] n[2] n[3] 0;
      0 0    0    0    1]

B = [0 1 0;
     α 0 β;
     0 γ 0]

λ = sqrt(β * γ + α)
@assert eigen(B).values ≈ λ * [-1,0,1]

eigen(B).values - [-1,0,1] * sqrt(β * γ + α)
v1 = [-1, λ, -γ] / sqrt(1 + λ^2 + γ^2)
v2 = [ β, 0, -α] / sqrt(β^2 + α^2)
v3 = [ 1, λ,  γ] / sqrt(1 + λ^2 + γ^2)

@assert min(norm(eigen(B).vectors[:, 1] - v1), norm(eigen(B).vectors[:, 1] + v1)) < 1e-15
@assert min(norm(eigen(B).vectors[:, 2] - v2), norm(eigen(B).vectors[:, 2] + v2)) < 1e-15
@assert min(norm(eigen(B).vectors[:, 3] - v3), norm(eigen(B).vectors[:, 3] + v3)) < 1e-15

V = [-1  β 1;
      λ  0 λ;
     -γ -α γ]

W = [-α / (2α + 2γ * β)   1 / 2λ  -β / (2α + 2γ * β);
     2γ / (2α + 2γ * β)   0       -2 / (2α + 2γ * β);
      α / (2α + 2γ * β)   1 / 2λ   β / (2α + 2γ * β)]

@assert norm(V * W - I) < 1e-15
