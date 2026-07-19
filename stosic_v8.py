from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v8.py — 7-node krug (K=7 / prilagodjenje 7/39) — Kantorovich duality (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/kantorovich_duality.pvs
  — dual potentials (ψ, φ); dual_feasible: φ(j)-ψ(i) ≤ c(i,j)
  — thm_dual_optimal: (ψ,φ) maksimizuje dual value
  — c = ||x-y||² u feature prostoru

Mapiranje na 7/39:
  x_k = ko-pojavljivanje na celom CSV (ℝ³⁹); c(i,j)=||x_i-x_j||²
  μ = poslednje izvlačenje; ν = empirija CSV
  entropski Kantorovich (Sinkhorn) → dual φ ~ ε log v
  next = top 7 po φ (dualna strana sertifikata)
  bez randoma; ε=SEED/100; odbaci 7 uzastopnih ako ikad
"""

from typing import List, Tuple

import numpy as np

from stosic_v1 import EPS, MAX_NUM, N_PICK, SEED, load_draws
from stosic_v2 import top7_from_freq

SINKHORN_ITERS = 200
SINKHORN_EPS = float(SEED) / 100.0


def measure_draw(draw: np.ndarray) -> np.ndarray:
    mu = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draw:
        mu[int(n) - 1] += 1.0 / N_PICK
    s = mu.sum()
    return mu / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)


def measure_empirical(draws: np.ndarray) -> np.ndarray:
    nu = np.zeros(MAX_NUM, dtype=np.float64)
    for d in draws:
        for n in d:
            nu[int(n) - 1] += 1.0
    return nu / nu.sum()


def cooccurrence_features(draws: np.ndarray) -> np.ndarray:
    co = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for d in draws:
        nums = [int(n) - 1 for n in d]
        for a in nums:
            for b in nums:
                co[a, b] += 1.0
    norms = np.linalg.norm(co, axis=1, keepdims=True)
    return co / np.maximum(norms, EPS)


def cost_matrix(features: np.ndarray) -> np.ndarray:
    sq = np.sum(features * features, axis=1)
    C = sq[:, None] + sq[None, :] - 2.0 * (features @ features.T)
    return np.maximum(C, 0.0)


def sinkhorn_dual(
    mu: np.ndarray, nu: np.ndarray, C: np.ndarray, eps: float, iters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Vraća (ψ, φ) aproksimaciju dualnih potencijala: ε log u, ε log v."""
    K = np.exp(-C / max(eps, EPS))
    u = np.ones(MAX_NUM, dtype=np.float64)
    v = np.ones(MAX_NUM, dtype=np.float64)
    mu = np.clip(mu, EPS, None)
    nu = np.clip(nu, EPS, None)
    mu = mu / mu.sum()
    nu = nu / nu.sum()
    for _ in range(iters):
        u = mu / np.maximum(K @ v, EPS)
        v = nu / np.maximum(K.T @ u, EPS)
    psi = eps * np.log(np.maximum(u, EPS))
    phi = eps * np.log(np.maximum(v, EPS))
    return psi, phi


def is_degenerate_consecutive(combo: List[int]) -> bool:
    s = sorted(combo)
    return (
        len(s) == N_PICK
        and s[-1] - s[0] == N_PICK - 1
        and s == list(range(s[0], s[0] + N_PICK))
    )


def predict_next(draws: np.ndarray) -> List[int]:
    C = cost_matrix(cooccurrence_features(draws))
    mu = measure_draw(draws[-1])
    nu = measure_empirical(draws)
    _psi, phi = sinkhorn_dual(mu, nu, C, SINKHORN_EPS, SINKHORN_ITERS)
    combo = top7_from_freq(phi)
    if is_degenerate_consecutive(combo):
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate_consecutive(next_combo):
        raise SystemExit("degenerisan next (uzastopni) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[5, x, 16, y, 29, z, 37]
"""



"""
v8: kantorovich_duality — Σ optimalnih π na uzastopnim kolima, skor iz last.

dual φ (Kantorovich) uz c=||x_i−x_j||² u ko-pojavljivanju; stop ako uzastopni.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
