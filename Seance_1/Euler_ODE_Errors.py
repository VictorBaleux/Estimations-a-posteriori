
"""
Euler_ODE_Errors.py — version mise à jour
----------------------------------------------
EDO: u'(t) = -λ u(t), u(0)=u0.

Modifs (sept. 2025) :
1) Quadrature L2 : rectangles à gauche remplacés par trapèze / Simpson (si N pair).
2) Stabilité Euler : assertion robuste — si λ=0 ne rien imposer ; sinon Δt ≤ (2−ε)/λ.
3) Choix des pas : prend Δt = T/N avec N entier (échelonné log) pour éviter un "dernier pas" irrégulier.

Figures générées (inchangées) :
1) Comparaison_visuelle.png
   -> 2×2 : haut Δt=1 s, bas Δt=0.001 s ; (gauche) solutions, (droite) erreur
2) Erreur_vs_delta_temps.png
   -> Erreurs L2 (u et u') en fonction de Δt (log-log)
3) Erreur_vs_derivé.png
   -> Scatter de l'erreur ponctuelle |e(t_n)| en fonction de la norme
      de la dérivée exacte |u'_{ex}(t_n)|, pour Δt=1 s et Δt=0.001 s.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class Problem:
    lam: float = 1.0   # λ
    u0: float = 1.0    # condition initiale
    T: float = 60.0    # horizon temporel (1 minute)


# ------------------------------
#   Intégrateur d'Euler explicite
# ------------------------------

def euler_explicite(pb: Problem, dt: Optional[float] = None, N: Optional[int] = None, eps: float = 1e-12
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intègre u' = -λ u par Euler explicite avec pas Δt = T/N (N entier).
    On peut fournir soit dt (approx.), soit directement N. Si dt est fourni,
    on prend N = round(T/dt) puis Δt := T/N pour tomber EXACTEMENT à T.

    Retourne:
        t : instants (taille N+1, réguliers de 0 à T)
        u : solution numérique aux instants t
        dts: tableau de taille N rempli par Δt (tous les pas identiques)
    """
    T = pb.T
    lam = pb.lam
    u0 = pb.u0

    if N is None:
        if dt is None:
            raise ValueError("Fournir soit dt, soit N.")
        N = max(1, int(round(T / float(dt))))
    else:
        N = int(N)
        if N <= 0:
            raise ValueError("N doit être un entier positif.")

    dt = T / N  # pas EXACT et régulier
    # Stabilité Euler robuste
    if lam != 0.0 and lam > 0.0:
        bound = (2.0 - eps) / lam
        assert dt <= bound, (
            f"Instable pour Euler explicite: λΔt={lam*dt:.3g} ; "
            f"attendu Δt ≤ (2−ε)/λ ≈ {bound:.6g} (ε={eps:g})."
        )
    # Si λ=0 : ne rien imposer.

    t = np.linspace(0.0, T, N + 1, dtype=float)
    u = np.empty(N + 1, dtype=float)
    u[0] = u0
    for n in range(N):
        u[n + 1] = u[n] + dt * (-lam * u[n])

    dts = np.full(N, dt, dtype=float)
    return t, u, dts


# ------------------------------
#   Outils exacts / quadrature
# ------------------------------

def u_exact(t: np.ndarray, pb: Problem) -> np.ndarray:
    return pb.u0 * np.exp(-pb.lam * t)


def _integrate_uniform(y: np.ndarray, dt: float) -> float:
    """
    Intègre une fonction tabulée y(t_n) sur [0,T] avec pas uniforme dt.
    Utilise Simpson si le nombre d'intervalles N=len(y)-1 est pair, sinon trapèze.
    Retourne l'intégrale numérique (pas la racine).
    """
    N = y.size - 1
    if N <= 0:
        return 0.0
    if N % 2 == 0 and N >= 2:
        # Simpson
        s_odd = np.sum(y[1:N:2])
        s_even = np.sum(y[2:N-1:2]) if N >= 3 else 0.0
        return (dt / 3.0) * (y[0] + y[-1] + 4.0 * s_odd + 2.0 * s_even)
    # Trapèze
    return dt * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def l2_error_function(t: np.ndarray, u_num: np.ndarray, dts: np.ndarray, pb: Problem) -> float:
    """
    ||e||_{L2(0,T)} ≈ ( ∫_0^T |u_num(t)-u_ex(t)|^2 dt )^{1/2}
    Quadrature: Simpson (si possible) sinon trapèze — pas uniforme Δt = dts[0].
    """
    dt = float(dts[0])
    e = u_num - u_exact(t, pb)
    integ = _integrate_uniform(e**2, dt)
    return float(np.sqrt(integ))


def l2_error_derivative(t: np.ndarray, u_num: np.ndarray, dts: np.ndarray, pb: Problem) -> float:
    """
    ||e'||_{L2(0,T)} ≈ ( ∫_0^T |u'_num(t) - u'_ex(t)|^2 dt )^{1/2}
    - u'_num(t_n) : dérivée numérique aux noeuds via différences centrales (np.gradient)
    - u'_ex(t_n)  : -λ u_ex(t_n)
    Quadrature: Simpson (si possible) sinon trapèze — pas uniforme.
    """
    dt = float(dts[0])
    uprime_num = np.gradient(u_num, dt)  # central diff interne, 1er ordre aux bords
    uprime_ex = -pb.lam * u_exact(t, pb)
    eprime = uprime_num - uprime_ex
    integ = _integrate_uniform(eprime**2, dt)
    return float(np.sqrt(integ))


def convergence_slope(dts: np.ndarray, errs: np.ndarray) -> float:
    x = np.log(dts)
    y = np.log(errs)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


# ------------------------------
#   Figures
# ------------------------------

def plot_part1_two_rows(pb: Problem,
                        dt_top: float = 1.0,
                        dt_bottom: float = 1e-3,
                        savepath: str = "Comparaison_visuelle.png") -> None:
    """
    Figure 2x2 : top = Δt=1 s ; bottom = Δt=0.001 s.
    Colonnes: (gauche) solutions ; (droite) erreur ponctuelle.
    Δt est projeté sur T/N (N entier) pour éviter un dernier pas irrégulier.
    """
    # TOP (Δt ≈ 1 s, mais forcé à T/N)
    t1, u1, dts1 = euler_explicite(pb, dt_top)
    ue1 = u_exact(t1, pb)
    err1 = np.abs(u1 - ue1)

    # BOTTOM (Δt ≈ 0.001 s, mais forcé à T/N)
    t2, u2, dts2 = euler_explicite(pb, dt_bottom)
    ue2 = u_exact(t2, pb)
    err2 = np.abs(u2 - ue2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax00, ax01), (ax10, ax11) = axes

    # Top-left: solutions Δt
    ax00.plot(t1, ue1, label="Solution exacte $u_{ex}(t)$")
    ax00.plot(t1, u1, marker="o", linestyle="--", label=fr"Euler $\Delta t={dts1[0]:g}$ s")
    ax00.set_xlabel("Temps t (s)")
    ax00.set_ylabel("Amplitude u(t)")
    ax00.set_title("Solutions — Δt ≈ 1 s")
    ax00.grid(True, alpha=0.3)
    ax00.legend()

    # Top-right: erreur Δt
    ax01.plot(t1, err1, marker="o", linestyle="-", label=r"$|e(t_n)|$")
    ax01.set_xlabel("Temps t (s)")
    ax01.set_ylabel("Erreur ponctuelle")
    ax01.set_title("Erreur — Δt ≈ 1 s")
    ax01.grid(True, alpha=0.3)
    ax01.legend()

    # Bottom-left: solutions Δt
    ax10.plot(t2, ue2, label="Solution exacte $u_{ex}(t)$")
    ax10.plot(t2, u2, linestyle="-", linewidth=1.0, label=fr"Euler $\Delta t={dts2[0]:g}$ s")
    ax10.set_xlabel("Temps t (s)")
    ax10.set_ylabel("Amplitude u(t)")
    ax10.set_title("Solutions — Δt ≈ 0.001 s")
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    # Bottom-right: erreur Δt
    ax11.plot(t2, err2, linestyle="-", linewidth=1.0, label=r"$|e(t_n)|$")
    ax11.set_xlabel("Temps t (s)")
    ax11.set_ylabel("Erreur ponctuelle")
    ax11.set_title("Erreur — Δt ≈ 0.001 s")
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    fig.suptitle("u'(t) = -λ u, u(0)=1 — Comparaison visuelle Δt", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=150)


def _logspace_integers(n_min: int, n_max: int, n_steps: int) -> np.ndarray:
    """Valeurs entières échelonnées logarithmiquement dans [n_min, n_max]."""
    vals = np.geomspace(max(1, n_min), max(1, n_max), max(2, n_steps))
    ints = np.unique(np.clip(np.round(vals).astype(int), n_min, n_max))
    return ints


def plot_part2(pb: Problem, n_steps: int = 20, dt_min: float = 1e-3, dt_max: float = 1.0,
               savepath: str = "Erreur_vs_delta_temps.png") -> None:
    """
    Δt définis via Δt = T/N avec N entier échelonné log entre
    N_min = ceil(T/dt_max) et N_max = floor(T/dt_min).
    Trace ||e||_L2 et ||e'||_L2 en fonction de Δt (log-log).
    """
    T = pb.T
    N_min = int(np.ceil(T / dt_max))
    N_max = int(np.floor(T / dt_min))
    if N_max < max(2, N_min):
        raise ValueError("Plage (dt_min, dt_max) trop étroite pour construire des N entiers.")
    N_list = _logspace_integers(N_min, N_max, n_steps)

    dts_arr = []
    errL2_u: List[float] = []
    errL2_du: List[float] = []

    for N in N_list:
        t, u_num, dts = euler_explicite(pb, N=N)
        dts_arr.append(float(dts[0]))
        errL2_u.append(l2_error_function(t, u_num, dts, pb))
        errL2_du.append(l2_error_derivative(t, u_num, dts, pb))

    dts_arr = np.array(dts_arr, dtype=float)
    errL2_u = np.array(errL2_u, dtype=float)
    errL2_du = np.array(errL2_du, dtype=float)

    slope_u = convergence_slope(dts_arr, errL2_u)
    slope_du = convergence_slope(dts_arr, errL2_du)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    ax.loglog(dts_arr, errL2_u, marker="o", linestyle="-", label=r"$\|u_h-u_{ex}\|_{L^2(0,T)}$")
    ax.loglog(dts_arr, errL2_du, marker="s", linestyle="--", label=r"$\|u'_h-u'_{ex}\|_{L^2(0,T)}$")
    ax.set_xlabel("Pas de temps Δt (s)")
    ax.set_ylabel("Erreur L2 sur [0, T]")
    ax.set_title(f"Erreurs L2 vs Δt — pentes ≈ {slope_u:.2f} (u), {slope_du:.2f} (u')")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)

    # Impression console (utile pour le rapport)
    print(f"Pente de convergence (log-log) pour ||u_h - u_ex||_L2 : {slope_u:.4f}")
    print(f"Pente de convergence (log-log) pour ||u'_h - u'_ex||_L2 : {slope_du:.4f}")


def plot_error_vs_exact_derivative(pb: Problem,
                                   dts_to_show: List[float] = [1.0, 1e-3],
                                   savepath: str = "Erreur_vs_derivé.png") -> None:
    """
    Scatter: erreur ponctuelle |e(t_n)| en fonction de |u'_{ex}(t_n)|,
    pour plusieurs pas de temps (par défaut: Δt=1 s et Δt=0.001 s).
    Deux sous-graphes: (gauche) axes linéaires, (droite) log-log.
    Δt est projeté sur T/N (N entier) pour éviter un dernier pas irrégulier.
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))

    for dt in dts_to_show:
        t, u_num, dts = euler_explicite(pb, dt)
        ue = u_exact(t, pb)
        err = np.abs(u_num - ue)
        deriv_norm = np.abs(-pb.lam * ue)  # = pb.lam * |ue|

        ax_lin.scatter(deriv_norm, err, s=10, alpha=0.6, label=fr"$\Delta t={dts[0]:g}$ s")
        ax_log.loglog(deriv_norm + 1e-16, err + 1e-16, marker="o", linestyle="", markersize=3, alpha=0.6, label=fr"$\Delta t={dts[0]:g}$ s")

    ax_lin.set_xlabel(r"$|u'_{ex}(t_n)|$")
    ax_lin.set_ylabel(r"$|e(t_n)|$")
    ax_lin.set_title("Erreur vs norme de la dérivée exacte (linéaire)")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend()

    ax_log.set_xlabel(r"$|u'_{ex}(t_n)|$")
    ax_log.set_ylabel(r"$|e(t_n)|$")
    ax_log.set_title("Erreur vs norme de la dérivée exacte (log-log)")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend()

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)


def main():
    pb = Problem(lam=1.0, u0=1.0, T=60.0)

    # Partie 1 : deux rangées: Δt ≈ 1 s (haut) et Δt ≈ 0.001 s (bas),
    # mais forcés à Δt = T/N exactement.
    plot_part1_two_rows(pb, dt_top=1.0, dt_bottom=1e-3,
                        savepath="Comparaison_visuelle.png")

    # Partie 2 : erreur L2 en fonction de Δt (N entiers échelonnés log entre dt_max et dt_min)
    plot_part2(pb, n_steps=20, dt_min=1e-3, dt_max=1.0,
               savepath="Erreur_vs_delta_temps.png")

    # Partie 3 : Erreur ponctuelle vs norme de la dérivée exacte
    plot_error_vs_exact_derivative(pb, dts_to_show=[1.0, 1e-3],
                                   savepath="Erreur_vs_derive.png")


if __name__ == "__main__":
    main()
