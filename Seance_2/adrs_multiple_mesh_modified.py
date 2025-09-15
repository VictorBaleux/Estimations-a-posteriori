# -*- coding: utf-8 -*-
"""
EF P1 (numpy/scipy) pour: V·∇u - div(ν∇u) + λ u = f  sur Ω=(0,L)^2, Dirichlet = u_exact (gaussienne)
- Convergence L2/H1 (6 maillages h=L/10..L/320)
- Fit (moindres carrés) sur  ||e||_L2 / |u|_H2  ~  C h^(k+1)
                         et  ||e||_H1 / |u|_H2  ~  C h^(k)
- Calcule M_h = ||u-uh||_L2 / ||u-Ph(u)||_L2
Dépendances: numpy, scipy (sparse, spsolve), matplotlib
"""

import numpy as np
import math
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ------------------ Paramètres physiques & numériques ------------------
L = 1.0
nu = 1e-1     # diffusivité
lam = 1.0     # réaction
beta = np.array([1.0, 0.5])  # convection constante (Vx, Vy)

N_list = [10, 20, 40, 80, 160, 320]  # subdivisions par direction
Nint_for_H2 = max(N_list)            # maillage d'intégration pour |u|_{H2}

# ------------------ u_exact = gaussienne -------------------------------
# u(x,y) = exp(-((x-x0)^2+(y-y0)^2)/(2*sigma^2))
x0, y0 = 0.5*L, 0.5*L
sigma = 0.15*L

def u_exact(x, y):
    r2 = (x - x0)**2 + (y - y0)**2
    return np.exp(-0.5 * r2 / (sigma**2))

def grad_u_exact(x, y):
    u = u_exact(x, y)
    ux = - (x - x0) / (sigma**2) * u
    uy = - (y - y0) / (sigma**2) * u
    return np.array([ux, uy])

def hess_u_exact(x, y):
    """Retourne u_xx, u_xy, u_yy (gaussienne)."""
    u = u_exact(x, y)
    dx = x - x0
    dy = y - y0
    s2 = sigma**2
    u_xx = (dx*dx / s2**2 - 1.0/s2) * u
    u_yy = (dy*dy / s2**2 - 1.0/s2) * u
    u_xy = (dx*dy / s2**2) * u
    return u_xx, u_xy, u_yy

def lap_u_exact(x, y):
    u_xx, u_xy, u_yy = hess_u_exact(x, y)
    return u_xx + u_yy

def f_exact(x, y):
    gx, gy = grad_u_exact(x, y)
    adv = beta[0]*gx + beta[1]*gy
    return adv - nu*lap_u_exact(x, y) + lam*u_exact(x, y)

# ------------------ Maillage structuré + triangulation P1 ---------------
def build_mesh(N):
    xs = np.linspace(0.0, L, N+1)
    ys = np.linspace(0.0, L, N+1)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):  # i: 0..N (x), j: 0..N (y)
        return i + j*(N+1)

    tris = []
    for j in range(N):
        for i in range(N):
            n00 = idx(i,   j)
            n10 = idx(i+1, j)
            n01 = idx(i,   j+1)
            n11 = idx(i+1, j+1)
            tris.append([n00, n10, n01])
            tris.append([n10, n11, n01])
    tris = np.array(tris, dtype=int)

    I, J = np.meshgrid(np.arange(N+1), np.arange(N+1), indexing='xy')
    bmask = (I==0) | (I==N) | (J==0) | (J==N)
    bmask = bmask.ravel()

    return nodes, tris, bmask

# ------------------ Outils P1 sur un triangle --------------------------
def tri_geom(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    As = 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))  # aire signée
    area = abs(As)
    denom = 2.0*As  # signé
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)
    grads = np.column_stack([b, c]) / denom  # (3,2): ∇φ_i
    return area, grads

def local_matrices(p1, p2, p3):
    area, grads = tri_geom(p1, p2, p3)

    # Diffusion
    G = grads @ grads.T
    Kdiff = nu * area * G

    # Masse (P1)
    Mloc = (area/12.0) * np.array([[2,1,1],
                                   [1,2,1],
                                   [1,1,2]], dtype=float)

    # Réaction
    Kreac = lam * Mloc

    # Advection (Galerkin): A_ij = (β·∇φ_j) * ∫ φ_i = (β·∇φ_j) * area/3
    s = grads @ beta  # s_j = β·∇φ_j
    Aadv = np.tile(s, (3,1)) * (area/3.0)

    return Kdiff, Aadv, Kreac, Mloc, area, grads

def triangle_quadrature_deg2(P):
    lambdas = np.array([[1/6, 1/6, 2/3],
                        [1/6, 2/3, 1/6],
                        [2/3, 1/6, 1/6]], dtype=float)
    X = lambdas @ P
    area, _ = tri_geom(*P)
    w = np.full(3, area/3.0)
    return X, w, lambdas

# ------------------ Assemblage système linéaire ------------------------
def assemble_system(N):
    nodes, tris, bmask = build_mesh(N)
    nn = nodes.shape[0]
    nt = tris.shape[0]

    rows = []; cols = []; vals = []
    rhs = np.zeros(nn)

    for t in range(nt):
        vid = tris[t]
        P = nodes[vid]
        Kdiff, Aadv, Kreac, Mloc, area, grads = local_matrices(*P)

        Aloc = Kdiff + Aadv + Kreac

        ii, jj = np.meshgrid(vid, vid, indexing='ij')
        rows.extend(ii.ravel()); cols.extend(jj.ravel()); vals.extend(Aloc.ravel())

        # Source f: quadrature à 3 points
        Xq, wq, lambdas = triangle_quadrature_deg2(P)
        fvals = np.array([f_exact(x, y) for x, y in Xq])
        for a in range(3):
            rhs[vid[a]] += np.dot(fvals, lambdas[:, a] * wq)

    A = coo_matrix((vals, (rows, cols)), shape=(nn, nn)).tocsr()

    # Dirichlet sur le bord: u = u_exact
    g = np.array([u_exact(x, y) for x, y in nodes])

    A = A.tolil()
    fixed = np.where(bmask)[0]
    for i in fixed:
        A.rows[i] = [i]
        A.data[i] = [1.0]
        rhs[i] = g[i]
    A = A.tocsr()

    uh = spsolve(A, rhs)
    return nodes, tris, uh, g, bmask

# ------------------ Normes, erreurs & |u|_{H2} ------------------------
def compute_errors(nodes, tris, uh):
    eL2 = 0.0
    eH1_semi = 0.0
    interpL2 = 0.0

    # Interpolant P1 nodal de u_exact
    uI = np.array([u_exact(x, y) for x, y in nodes])

    for t in range(tris.shape[0]):
        vid = tris[t]
        P = nodes[vid]
        Xq, wq, lambdas = triangle_quadrature_deg2(P)

        # L2
        uh_loc = uh[vid]
        uh_q = lambdas @ uh_loc
        u_q = np.array([u_exact(x, y) for x, y in Xq])
        e_q = u_q - uh_q
        eL2 += np.dot(wq, e_q**2)

        # L2 interpolation
        uI_q = lambdas @ uI[vid]
        ei_q = u_q - uI_q
        interpL2 += np.dot(wq, ei_q**2)

        # H1-semi: ∫ ||∇u - ∇uh||^2
        area, grads = tri_geom(*P)
        grad_uh = grads.T @ uh_loc  # constant sur l'élément
        # quadrature à 3 pts pour ∇u
        val = 0.0
        for q in range(3):
            gu = grad_u_exact(Xq[q,0], Xq[q,1])
            d = gu - grad_uh
            val += wq[q] * (d[0]**2 + d[1]**2)
        eH1_semi += val

    eL2 = math.sqrt(eL2)
    eH1_semi = math.sqrt(eH1_semi)
    eH1 = math.sqrt(eL2**2 + eH1_semi**2)
    interpL2 = math.sqrt(interpL2)
    return eL2, eH1_semi, eH1, interpL2

def h2_seminorm_exact(N):
    """Calcule |u|_{H^2} via quadrature sur un maillage N x N.
    Convention: |u|_{H2}^2 = ∫ (u_xx^2 + 2 u_xy^2 + u_yy^2) dx
    """
    nodes, tris, _ = build_mesh(N)
    val = 0.0
    for t in range(tris.shape[0]):
        P = nodes[tris[t]]
        Xq, wq, _ = triangle_quadrature_deg2(P)
        for q in range(3):
            u_xx, u_xy, u_yy = hess_u_exact(Xq[q,0], Xq[q,1])
            val += wq[q] * (u_xx**2 + 2.0*(u_xy**2) + u_yy**2)
    return math.sqrt(val)

# ------------------ Étude de convergence -------------------------------
print("Calcul de |u|_{H2} (gaussienne) ...")
H2_semi_u = h2_seminorm_exact(Nint_for_H2)
print(f"|u|_H2 ≈ {H2_semi_u:.8e} (maillage Nint={Nint_for_H2})")

results = []
for N in N_list:
    print(f"Assemblage/résolution pour N={N} ...")
    nodes, tris, uh, g, bmask = assemble_system(N)
    eL2, eH1s, eH1, iL2 = compute_errors(nodes, tris, uh)
    h = L / N
    Mh = eL2 / iL2 if iL2 > 0 else np.nan
    results.append((h, eL2, eH1s, eH1, iL2, Mh))

# Trier par h croissant
results.sort(key=lambda r: r[0])

# Impression tableau
print("\n=== Tableau des erreurs (gaussienne) et M_h ===")
print("     h        ||e||_L2        |e|_H1        ||e||_H1     ||u-Ph(u)||_L2      M_h")
for h, e2, e1s, e1, i2, Mh in results:
    print(f"{h:8.5f}  {e2:12.5e}  {e1s:12.5e}  {e1:12.5e}   {i2:14.5e}   {Mh:7.3f}")

# ------------------ Fit sur erreurs NORMALISÉES par |u|_H2 -------------
def fit_power(h_arr, err_arr):
    X = np.log(h_arr); Y = np.log(err_arr)
    p, logC = np.polyfit(X, Y, 1)  # Y = logC + p log h
    return float(np.exp(logC)), float(p)

h_vals = np.array([r[0] for r in results])
err_L2 = np.array([r[1] for r in results])
err_H1 = np.array([r[3] for r in results])

# Normalisation par |u|_{H2}
err_L2_norm = err_L2 / H2_semi_u
err_H1_norm = err_H1 / H2_semi_u

C_L2n, p_L2n = fit_power(h_vals, err_L2_norm)  # p ≈ k+1 (P1 => ≈2)
C_H1n, p_H1n = fit_power(h_vals, err_H1_norm)  # p ≈ k   (P1 => ≈1)

k_from_L2 = p_L2n - 1.0
k_from_H1 = p_H1n

print("\n=== Ajustements (moindres carrés) sur erreurs normalisées par |u|_H2 ===")
print("Formes ajustées :  ||e||_L2 / |u|_H2 ≈ C_L2 h^(k+1)   et   ||e||_H1 / |u|_H2 ≈ C_H1 h^k")
print(f"L2  : pente p={p_L2n:.4f}  => k≈{k_from_L2:.4f},   C_L2≈{C_L2n:.4e}")
print(f"H1  : pente p={p_H1n:.4f}  => k≈{k_from_H1:.4f},   C_H1≈{C_H1n:.4e}")

# ------------------ Figures log-log (erreurs normalisées) --------------
plt.figure(figsize=(6,5))
plt.loglog(h_vals, err_L2_norm, 'o-', label=r"$\|u-u_h\|_{L^2}/|u|_{H^2}$")
plt.loglog(h_vals, err_H1_norm, 's-', label=r"$\|u-u_h\|_{H^1}/|u|_{H^2}$")

hh = np.linspace(h_vals.min(), h_vals.max(), 200)
plt.loglog(hh, C_L2n * hh**p_L2n, '--', label=fr"fit L2: $h^{{{p_L2n:.2f}}}$")
plt.loglog(hh, C_H1n * hh**p_H1n, '--', label=fr"fit H1: $h^{{{p_H1n:.2f}}}$")

plt.gca().invert_xaxis()
plt.grid(True, which="both", ls=":")
plt.xlabel("h")
plt.ylabel("Erreur normalisée")
plt.title("Convergence EF P1 — Gaussienne — erreurs normalisées par |u|_{H^2}")
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png", dpi=200)  
plt.show()

# ------------------ Commentaires ------------------------------
print("\nCommentaires:")
print("- Avec P1 non stabilisé et ν modéré, on s’attend à p≈2 en L2 et p≈1 en H1,")
print("  donc k≈1. Le fit est effectué sur les erreurs normalisées par |u|_{H^2},")
print("  ce qui isole la constante C liée au schéma (et au problème) du facteur |u|_{H^2}.")
print("- La constante M_h = ||u-uh||_L2 / ||u-Ph(u)||_L2 est également reportée.")
