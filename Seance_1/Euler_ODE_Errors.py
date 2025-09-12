"""
Euler_ODE_Errors.py
----------------------------------------------
EDO: u'(t) = -λ u(t), u(0)=u0, λ=1.

Figures générées :
1) Comparaison_visuelle.png
   -> 2×2 : haut Δt=1 s, bas Δt=0.001 s ; (gauche) solutions, (droite) erreur
2) Erreur_vs_delta_temps.png
   -> Erreurs L2 (u et u') en fonction de Δt (log-log)
3) Erreur_vs_derivé.png
   -> Scatter de l'erreur ponctuelle |e(t_n)| en fonction de la norme
      de la dérivée exacte |u'_{ex}(t_n)|, pour Δt=1 s et Δt=0.001 s.
      Sous-figure gauche: axes linéaires ; droite: log-log.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Problem:
    lam: float = 1.0   # λ
    u0: float = 1.0    # condition initiale
    T: float = 60.0    # horizon temporel (1 minute)


def euler_explicite(pb: Problem, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intègre u' = -λ u par Euler explicite avec pas nominal dt jusqu'à T.
    Le dernier pas est ajusté (si nécessaire) pour tomber exactement à T.
    Retourne:
        t : instants (taille N+1)
        u : solution numérique aux instants t
        dts: taille de chaque intervalle (longueur N), avec dernier dt possiblement ajusté
    """
    T = pb.T
    lam = pb.lam
    u0 = pb.u0

    N_full = int(np.floor(T / dt))
    t_list = [0.0]
    u_list = [u0]
    
    assert dt < 2.0/lam, f"Instable pour Euler explicite: λΔt={lam*dt:.3g} (attendu < 2)."

    # Pas réguliers de taille dt
    for _ in range(N_full):
        un = u_list[-1]
        un1 = un + dt * (-lam * un)
        u_list.append(un1)
        t_list.append(t_list[-1] + dt)

    # Dernier pas ajusté pour atteindre exactement T (si besoin)
    t_current = t_list[-1]
    dt_last = T - t_current
    dts = [dt] * N_full  # liste des pas
    if dt_last > 1e-14:
        un = u_list[-1]
        un1 = un + dt_last * (-lam * un)
        u_list.append(un1)
        t_list.append(T)
        dts.append(dt_last)

    t = np.array(t_list, dtype=float)
    u = np.array(u_list, dtype=float)
    dts = np.array(dts, dtype=float)
    return t, u, dts


def u_exact(t: np.ndarray, pb: Problem) -> np.ndarray:
    return pb.u0 * np.exp(-pb.lam * t)


def l2_error_function(t: np.ndarray, u_num: np.ndarray, dts: np.ndarray, pb: Problem) -> float:
    ue = u_exact(t, pb)
    e_left = u_num[:-1] - ue[:-1]     # erreur évaluée au bord gauche de chaque intervalle
    return float(np.sqrt(np.sum((e_left**2) * dts)))


def l2_error_derivative(t: np.ndarray, u_num: np.ndarray, dts: np.ndarray, pb: Problem) -> float:
    # dérivée numérique par intervalle
    du = np.diff(u_num)
    uprime_num = du / dts  # taille N
    # points milieux de chaque intervalle
    t_mid = t[:-1] + 0.5 * dts
    # dérivée exacte aux milieux
    uprime_ex = -pb.lam * u_exact(t_mid, pb)
    eprime = uprime_num - uprime_ex
    return float(np.sqrt(np.sum((eprime**2) * dts)))


def convergence_slope(dts: np.ndarray, errs: np.ndarray) -> float:
    x = np.log(dts)
    y = np.log(errs)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def plot_part1_two_rows(pb: Problem,
                        dt_top: float = 1.0,
                        dt_bottom: float = 1e-3,
                        savepath: str = "Visual_comparison.png") -> None:
    """
    Figure 2x2 : top = Δt=1 s ; bottom = Δt=0.001 s.
    Colonnes: (gauche) solutions ; (droite) erreur ponctuelle.
    """
    # TOP (Δt = 1 s)
    t1, u1, dts1 = euler_explicite(pb, dt_top)
    ue1 = u_exact(t1, pb)
    err1 = np.abs(u1 - ue1)

    # BOTTOM (Δt = 0.001 s)
    t2, u2, dts2 = euler_explicite(pb, dt_bottom)
    ue2 = u_exact(t2, pb)
    err2 = np.abs(u2 - ue2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax00, ax01), (ax10, ax11) = axes

    # Top-left: solutions Δt=1 s
    ax00.plot(t1, ue1, label="Solution exacte $u_{ex}(t)$")
    ax00.plot(t1, u1, marker="o", linestyle="--", label=r"Euler $\Delta t=1$ s")
    ax00.set_xlabel("Temps t (s)")
    ax00.set_ylabel("Amplitude u(t)")
    ax00.set_title("Solutions — Δt = 1 s")
    ax00.grid(True, alpha=0.3)
    ax00.legend()

    # Top-right: erreur Δt=1 s
    ax01.plot(t1, err1, marker="o", linestyle="-", label=r"$|e(t_n)|$")
    ax01.set_xlabel("Temps t (s)")
    ax01.set_ylabel("Erreur ponctuelle")
    ax01.set_title("Erreur — Δt = 1 s")
    ax01.grid(True, alpha=0.3)
    ax01.legend()

    # Bottom-left: solutions Δt=0.001 s
    ax10.plot(t2, ue2, label="Solution exacte $u_{ex}(t)$")
    ax10.plot(t2, u2, linestyle="-", linewidth=1.0, label=r"Euler $\Delta t=0.001$ s")
    ax10.set_xlabel("Temps t (s)")
    ax10.set_ylabel("Amplitude u(t)")
    ax10.set_title("Solutions — Δt = 0.001 s")
    ax10.grid(True, alpha=0.3)
    ax10.legend()

    # Bottom-right: erreur Δt=0.001 s
    ax11.plot(t2, err2, linestyle="-", linewidth=1.0, label=r"$|e(t_n)|$")
    ax11.set_xlabel("Temps t (s)")
    ax11.set_ylabel("Erreur ponctuelle")
    ax11.set_title("Erreur — Δt = 0.001 s")
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    fig.suptitle("u'(t) = -λ u, u(0)=1 ; λ=1 — Comparaison visuelle Δt", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=150)


def plot_part2(pb: Problem, n_steps: int = 20, dt_min: float = 1e-3, dt_max: float = 1.0,
               savepath: str = "L2_error_vs_deta_time.png") -> None:
    """
    Δt décroissants de 1 s à 0.001 s (échelle logarithmique), 20 valeurs.
    Trace ||e||_L2 et ||e'||_L2 en fonction de Δt (log-log).
    """
    dts_list = np.logspace(np.log10(dt_max), np.log10(dt_min), n_steps)
    errL2_u: List[float] = []
    errL2_du: List[float] = []

    for dt in dts_list:
        t, u_num, dts = euler_explicite(pb, dt)
        errL2_u.append(l2_error_function(t, u_num, dts, pb))
        errL2_du.append(l2_error_derivative(t, u_num, dts, pb))

    dts_arr = np.array(dts_list, dtype=float)
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
                                   savepath: str = "Error_vs_exact_derivative.png") -> None:
    """
    Scatter: erreur ponctuelle |e(t_n)| en fonction de |u'_{ex}(t_n)|,
    pour plusieurs pas de temps (par défaut: Δt=1 s et Δt=0.001 s).
    Deux sous-graphes: (gauche) axes linéaires, (droite) log-log.
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))

    for dt in dts_to_show:
        t, u_num, dts = euler_explicite(pb, dt)
        ue = u_exact(t, pb)
        err = np.abs(u_num - ue)
        deriv_norm = np.abs(-pb.lam * ue)  # = pb.lam * |ue|

        ax_lin.scatter(deriv_norm, err, s=10, alpha=0.6, label=fr"$\Delta t={dt:g}$ s")
        ax_log.loglog(deriv_norm + 1e-16, err + 1e-16, marker="o", linestyle="", markersize=3, alpha=0.6, label=fr"$\Delta t={dt:g}$ s")

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

    # Partie 1 : deux rangées: Δt = 1 s (haut) et Δt = 0.001 s (bas)
    plot_part1_two_rows(pb, dt_top=1.0, dt_bottom=1e-3,
                        savepath="Comparaison_visuelle.png")

    # Partie 2 : erreur L2 en fonction de Δt (20 valeurs entre 1 et 1e-3)
    plot_part2(pb, n_steps=20, dt_min=1e-3, dt_max=1.0,
               savepath="Erreur_vs_delta_temps.png")

    # Partie 3 : Erreur ponctuelle vs norme de la dérivée exacte
    plot_error_vs_exact_derivative(pb, dts_to_show=[1.0, 1e-3],
                                   savepath="Erreur_vs_derivé.png")


if __name__ == "__main__":
    main()
