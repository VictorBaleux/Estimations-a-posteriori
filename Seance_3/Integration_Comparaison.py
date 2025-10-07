#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison de méthodes d’intégration et étude de convergence (log–log + régressions).
- Riemann (points milieux) — maillage en x de taille N
- Lebesgue uniforme — maillage en y de taille M
- Lebesgue non uniforme (itératif) — maillage en y de taille N_it (évolutif)

Ajouts récents :
- Intégrale de référence via Simpson adaptatif (I_ref)
- Graphiques log–log (avec et sans droites de régression)
- Régression linéaire sur log10(points) vs log10(erreur) et affichage des pentes en console
- **Sur-raffinement** : à chaque itération, on **double** le nombre de découpes créées
  (k = ceil( overrefine_factor * count / capacity ), avec overrefine_factor=2 par défaut)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Callable, List, Tuple

# ----- Problème -----
Left, Right = 0.0, 1.0
a, b, c = 0.5, 10.0, 3.0

def f(x: np.ndarray) -> np.ndarray:
    return a*x**2 + b*x + c*np.sin(4*np.pi*x) + 10*np.exp(-100*(x-0.5)**2)

# -------------------------------------------------
# Référence : Simpson adaptatif sur [Left, Right]
# -------------------------------------------------
def _simpson(f, a, b):
    c = 0.5*(a+b)
    h = b - a
    return (h/6.0)*(f(a) + 4.0*f(c) + f(b))

def _adaptive_simpson(f, a, b, eps, whole, depth, max_depth):
    c = 0.5*(a+b)
    left = _simpson(f, a, c)
    right = _simpson(f, c, b)
    if depth >= max_depth:
        return left + right
    if abs(left + right - whole) < 15*eps:
        return left + right + (left + right - whole)/15.0
    return (_adaptive_simpson(f, a, c, eps/2.0, left, depth+1, max_depth) +
            _adaptive_simpson(f, c, b, eps/2.0, right, depth+1, max_depth))

def integral_reference(func: Callable[[float], float],
                       a: float, b: float, eps: float = 1e-12, max_depth: int = 30) -> float:
    whole = _simpson(func, a, b)
    return _adaptive_simpson(func, a, b, eps, whole, 0, max_depth)

# -------------------------
# Riemann (midpoint)
# -------------------------
def riemann_sum_midpoint(func: Callable[[np.ndarray], np.ndarray],
                         left: float, right: float, N: int) -> float:
    h = (right - left) / N
    x = left + h*(np.arange(N) + 0.5)
    return h * np.sum(func(x))

def riemann_convergence(func, left, right, N0=25, levels=8, I_ref=None):
    Ns = [N0 * (2**k) for k in range(levels)]
    Is = np.array([riemann_sum_midpoint(func, left, right, N) for N in Ns], dtype=float)
    if I_ref is None:
        abs_err = np.empty_like(Is); abs_err[:] = np.nan
    else:
        abs_err = np.abs(Is - I_ref)
    return np.array(Ns), Is, abs_err

# --------------------------
# Lebesgue (uniform en y)
# --------------------------
def y_bounds(func, left, right, Nx=100000, pad_ratio=0.05):
    x = np.linspace(left, right, Nx)
    y = func(x)
    y_min = y.min(); y_max = y.max()
    pad = pad_ratio * max(1.0, (y_max - y_min))
    return (y_min - pad), (y_max + pad), x, y

def lebesgue_integral_from_samples(y_vals: np.ndarray, left: float, right: float, 
                                   y_min: float, y_max: float, M: int):
    Nxs = y_vals.size
    L_edges = np.linspace(y_min, y_max, M+1)
    counts, _ = np.histogram(y_vals, bins=L_edges)
    measures = counts * (right - left) / float(Nxs)
    IL = np.sum(measures * L_edges[:-1])
    return IL, measures, L_edges, counts

def lebesgue_convergence(func, left, right, Nx=100000, M0=25, levels=8, I_ref=None):
    y_min, y_max, x_grid, y_vals = y_bounds(func, left, right, Nx=Nx)
    Ms = [M0 * (2**k) for k in range(levels)]
    ILs = []
    for M in Ms:
        IL, measures, L_edges, counts = lebesgue_integral_from_samples(y_vals, left, right, y_min, y_max, M)
        ILs.append(IL)
    ILs = np.array(ILs, dtype=float)
    if I_ref is None:
        abs_err = np.full_like(ILs, np.nan)
    else:
        abs_err = np.abs(ILs - I_ref)
    return (np.array(Ms), ILs, abs_err, y_min, y_max, x_grid, y_vals, Nx)

# -------------------------------------------------------------
# Lebesgue (non-uniforme en y) – VERSION ITÉRATIVE avec sur-raffinement
# -------------------------------------------------------------
def refine_bins_by_load(edges: np.ndarray, counts: np.ndarray, Nx_total: int, overrefine_factor: float = 2.0) -> np.ndarray:
    """
    Raffine chaque maille dont le chargement dépasse 1/N en la découpant
    en k sous-mailles : k = ceil( overrefine_factor * count / capacity ),
    où capacity = Nx_total / N. Par défaut, overrefine_factor=2 => on *double*
    les découpes nécessaires.
    """
    N = len(edges) - 1
    if N <= 0:
        return edges.copy()
    capacity = Nx_total / float(N)  # seuil: 1/N en proportion des N_x
    new_edges: List[float] = [float(edges[0])]
    for i in range(N):
        left = float(edges[i]); right = float(edges[i+1])
        ci = float(counts[i])
        if ci > capacity:  # surcharge
            k = int(np.ceil(overrefine_factor * ci / capacity))
            k = max(1, k)
            sub_edges = np.linspace(left, right, k+1)[1:]  # k sous-intervalles
            new_edges.extend(list(sub_edges))
        else:
            new_edges.append(right)
    return np.array(new_edges, dtype=float)

def lebesgue_nonuniform_iterative_refinement(
    func: Callable[[np.ndarray], np.ndarray],
    left: float, right: float,
    Nx: int = 100000,
    N0: int = 25,
    n_iters: int = 10,
    pad_ratio: float = 0.05,
    save_prefix: str = "lebesgue_nonuni_iter",
    show_plots: bool = True,
    I_ref: float | None = None,
    overrefine_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    # Échantillonnage
    x_grid = np.linspace(left, right, Nx)
    y_vals = func(x_grid)
    y_min = float(np.min(y_vals)); y_max = float(np.max(y_vals))
    pad = pad_ratio * max(1.0, (y_max - y_min))
    y_min -= pad; y_max += pad

    # Maillage initial (uniforme)
    edges = np.linspace(y_min, y_max, N0 + 1)
    Nx_total = len(y_vals)

    Ns, ILs = [], []
    abs_err = []

    edges_hist: List[np.ndarray] = []
    counts_hist: List[np.ndarray] = []

    def integral_from_edges(edges_arr: np.ndarray) -> Tuple[float, np.ndarray]:
        counts, _ = np.histogram(y_vals, bins=edges_arr)
        measures = counts * (right - left) / float(Nx_total)
        IL = float(np.sum(measures * edges_arr[:-1]))  # poids à gauche
        return IL, counts

    

    # Itération 0
    IL0, counts = integral_from_edges(edges)
    Ns.append(len(edges) - 1); ILs.append(IL0)
    abs_err.append(np.nan if I_ref is None else abs(IL0 - I_ref))
    edges_hist.append(edges.copy()); counts_hist.append(counts.copy())
    

    # Boucle d'affinage
    for it in range(1, n_iters + 1):
        new_edges = refine_bins_by_load(edges, counts, Nx_total, overrefine_factor=overrefine_factor)
        IL_next, counts_next = integral_from_edges(new_edges)

        Ns.append(len(new_edges) - 1); ILs.append(IL_next)
        abs_err.append(np.nan if I_ref is None else abs(IL_next - I_ref))

        edges_hist.append(new_edges.copy()); counts_hist.append(counts_next.copy())
        

        edges = new_edges; counts = counts_next

    Ns = np.array(Ns, dtype=int)
    ILs = np.array(ILs, dtype=float)
    abs_err = np.array(abs_err, dtype=float)

    # Sauvegardes convergence
    df = pd.DataFrame({"iteration": np.arange(len(Ns)), "N_bins": Ns, "IL": ILs, "abs_err": abs_err})
    df.to_csv(f"{save_prefix}_convergence.csv", index=False)

    return Ns, ILs, abs_err, edges_hist

# --------------------------
# Utilitaires graphiques & régressions
# --------------------------
def save_show(fig_path: str, show: bool):
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def regression_loglog(points: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """Régression linéaire sur (log10(points), log10(errors)). Retourne (pente, intercept)."""
    mask = (points > 0) & np.isfinite(points) & (errors > 0) & np.isfinite(errors)
    x = np.log10(points[mask])
    y = np.log10(errors[mask])
    if x.size < 2:
        return np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

# --------------------------
# Main
# --------------------------
def main(show_plots: bool = True):
    # f(x)
    x_plot = np.linspace(Left, Right, 2000)
    y_plot = f(x_plot)
    plt.figure()
    plt.plot(x_plot, y_plot, label="f(x)")
    plt.title("f(x) sur [0,1]")
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.legend()
    save_show("f_function_plot.png", show_plots)

    # Intégrale de référence
    I_ref = integral_reference(lambda xx: float(f(np.array([xx]))[0]), Left, Right, eps=1e-12, max_depth=30)
    with open("reference_integral.txt", "w", encoding="utf-8") as fh:
        fh.write(f"I_ref (Simpson adaptatif) = {I_ref:.16f}\n")

    # Convergences
    Ns_riem, Is_riem, err_riem = riemann_convergence(f, Left, Right, N0=25, levels=8, I_ref=I_ref)
    Ms_lebU, ILs_lebU, err_lebU, y_min, y_max, x_grid, y_vals, Nx_used = lebesgue_convergence(
        f, Left, Right, Nx=100000, M0=25, levels=8, I_ref=I_ref
    )
    Ns_iter, ILs_iter, err_iter, edges_hist = lebesgue_nonuniform_iterative_refinement(
        f, Left, Right, Nx=100000, N0=100, n_iters=6, show_plots=show_plots, I_ref=I_ref, overrefine_factor=2.0
    )

    # CSV consolidé
    rows = []
    for n, e in zip(Ns_riem, err_riem):
        rows.append({"method": "Riemann (milieux, x)", "points": int(n), "abs_error": float(e)})
    for m, e in zip(Ms_lebU, err_lebU):
        rows.append({"method": "Lebesgue uniforme (y)", "points": int(m), "abs_error": float(e)})
    for n, e in zip(Ns_iter, err_iter):
        rows.append({"method": "Lebesgue non uniforme (y, itératif, x2)", "points": int(n), "abs_error": float(e)})
    df_all = pd.DataFrame(rows)
    df_all.to_csv("errors_vs_points_all_methods.csv", index=False)

    # Régressions log–log (pentes)
    slope_R, bR = regression_loglog(Ns_riem, err_riem)
    slope_U, bU = regression_loglog(Ms_lebU, err_lebU)
    slope_I, bI = regression_loglog(Ns_iter, err_iter)

    print(f"I_ref (Simpson adaptatif) = {I_ref:.16f}")
    print("Pentes (log10-log10) estimées par régression linéaire :")
    print(f"  - Riemann (milieux, x)                 : pente ≈ {slope_R: .3f}")
    print(f"  - Lebesgue uniforme (y)                : pente ≈ {slope_U: .3f}")
    print(f"  - Lebesgue non uniforme (itér., x2)    : pente ≈ {slope_I: .3f}")
    

    # --------- Graphe log–log AVEC droites de régression ---------
    plt.figure()
    # points
    plt.loglog(Ns_riem, err_riem, marker="o", linestyle="None", label="Riemann (x)")
    plt.loglog(Ms_lebU, err_lebU, marker="s", linestyle="None", label="Lebesgue uni. (y)")
    plt.loglog(Ns_iter, err_iter, marker="^", linestyle="None", label="Lebesgue non uni. (itér., x2)")
    # droites ajustées
    def add_fit_line(points, slope, intercept):
        mask = (points > 0) & np.isfinite(points)
        if np.count_nonzero(mask) < 2 or not np.isfinite(slope):
            return
        x_min, x_max = points[mask].min(), points[mask].max()
        xs = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        ys = 10**(slope*np.log10(xs) + intercept)
        plt.loglog(xs, ys, linestyle="-")
    add_fit_line(Ns_riem, slope_R, bR)
    add_fit_line(Ms_lebU, slope_U, bU)
    add_fit_line(Ns_iter, slope_I, bI)
   
    plt.title("Erreur absolue vs nombre de points (log–log) — AVEC régressions")
    plt.xlabel("Nombre de points du maillage")
    plt.ylabel("Erreur absolue |I - I_ref|")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    save_show("errors_vs_points_all_methods_loglog_with_fit.png", show_plots)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(show_plots=not args.no_show)