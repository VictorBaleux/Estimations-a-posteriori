# adrs_multiple_mesh.py — version corrigée et augmentée
# Objectif: estimer l'ordre en espace du schéma P1 pour ADRS 1D,
#           calculer les erreurs L2/H1, la semi-norme H2 exacte,
#           normaliser, identifier (C,k) et tracer les superpositions.

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# PHYSICAL PARAMETERS
# ---------------------------
K = 0.1       # diffusion
L = 1.0       # taille du domaine
Time = 20.0   # temps max d'intégration (pour le pseudo-temps)
V = 1.0       # advection
lamda = 1.0   # reaction

# ---------------------------
# NUMERICAL PARAMETERS
# ---------------------------
NX0 = 21            # nombre de points de grille de départ
NT = 10000          # itérations de pseudo-temps max
ifre = 10**9        # affichage intermédiaire (très grand => quasi désactivé)
eps = 1e-3          # critère de convergence relatif (res/res0 <= eps)
niter_refinement = 10  # nombre de maillages (affinements successifs)
dNX = 5             # incrément du nombre de points par raffinement

# tableaux de stockage (taille niter_refinement)
errorL2 = np.zeros(niter_refinement)     # ||u-uh||_{L2}^2 (discret)
errorH1 = np.zeros(niter_refinement)     # ||u-uh||_{H1}^2 (discret) = L2 + semi-H1
semiH2  = np.zeros(niter_refinement)     # |u|_{H2}^2 (discret, sur la solution exacte)
itertab = np.zeros(niter_refinement)     # pas h=dx

# ---------------------------
# BOUCLE SUR LES MAILLAGES
# ---------------------------
NX = NX0
for it in range(niter_refinement):
    NX += dNX
    x = np.linspace(0.0, L, NX)
    dx = L / (NX - 1)
    # dt de stabilité (CFL diffusif/advection), proche de ce que tu avais
    dt = dx**2 / (abs(V)*dx + 4.0*K + dx**2)

    itertab[it] = dx
    print(f"[mesh {it+1}/{niter_refinement}] NX={NX}, dx={dx:.4e}, dt={dt:.4e}")

    # --- Initialisation champs ---
    T   = np.zeros(NX)     # solution approchée (initialisée à 0)
    RHS = np.zeros(NX)
    F   = np.zeros(NX)
    Tex = np.zeros(NX)     # solution exacte discrétisée sur la grille
    rest = []              # historique du résidu

    # --- Solution exacte: gaussienne centrée en 0.5 ---
    for j in range(1, NX-1):
        Tex[j] = np.exp(-20.0 * (j*dx - 0.5)**2)

    # --- Forçage F pour que Tex soit stationnaire ---
    # u_t = -V u_x + K u_xx - lamda u + F = 0  => F = V u_x - K u_xx + lamda u
    for j in range(1, NX-1):
        Tex_x  = (Tex[j+1] - Tex[j-1]) / (2.0*dx)           # dérivée 1 exacte (discrète)
        Tex_xx = (Tex[j+1] - 2.0*Tex[j] + Tex[j-1]) / dx**2 # dérivée 2 exacte (discrète)
        F[j] = V*Tex_x - K*Tex_xx + lamda*Tex[j]

    # ---------------------------
    # BOUCLE EN TEMPS (pseudo-stationnaire)
    # ---------------------------
    n  = 0
    res = res0 = 1.0

    while (n < NT) and (res/res0 > eps):
        n += 1
        res = 0.0

        # schéma explicite simple (mêmes expressions que ton code d'origine)
        for j in range(1, NX-1):
            xnu = K + 0.5*dx*abs(V)  # viscosité numérique (stabilisation)
            T_x  = (T[j+1] - T[j-1]) / (2.0*dx)
            T_xx = (T[j-1] - 2.0*T[j] + T[j+1]) / dx**2  # NB: (j-1 - 2j + j+1)
            RHS[j] = dt * (-V*T_x + xnu*T_xx - lamda*T[j] + F[j])
            res += abs(RHS[j])

        # mise à jour d'Euler
        for j in range(1, NX-1):
            T[j] += RHS[j]
            RHS[j] = 0.0

        if n == 1:
            res0 = max(res, 1e-16)

        rest.append(res)

        # affichage optionnel (désactivé par ifre très grand)
        if (n % ifre == 0) or (res/res0 <= eps):
            print(f"  it_time={n:5d}, res={res:.3e}, res/res0={res/res0:.3e}")

    print(f"  stop: it_time={n}, res={res:.3e}, res/res0={res/res0:.3e}")

    # ---------------------------
    # CALCUL DES ERREURS DISCRÈTES
    # ---------------------------
    errL2h = 0.0
    errH1_semi = 0.0
    semiH2_exact = 0.0

    for j in range(1, NX-1):
        # erreur L2
        errL2h += dx * (T[j] - Tex[j])**2

        # semi-norme H1 (erreur de dérivée 1)
        Tex_x  = (Tex[j+1] - Tex[j-1]) / (2.0*dx)   # dérivée 1 de l'exacte
        T_x    = (T[j+1]   - T[j-1])   / (2.0*dx)   # dérivée 1 de l'approchée
        errH1_semi += dx * (T_x - Tex_x)**2

        # semi-norme H2 de l'exacte (pour normalisation)
        Tex_xx = (Tex[j+1] - 2.0*Tex[j] + Tex[j-1]) / dx**2
        semiH2_exact += dx * (Tex_xx**2)

    errorL2[it] = errL2h
    errorH1[it] = errL2h + errH1_semi           # norme H1^2 = L2^2 + |.|_{H1}^2
    semiH2[it]  = semiH2_exact

# ---------------------------
# POST-TRAITEMENT & IDENTIFICATION (C,k)
# ---------------------------
h = itertab.copy()

# Normalisation correcte : diviser par sqrt(|u|_{H2}^2) = |u|_{H2}
E_L2 = np.sqrt(errorL2) / np.sqrt(semiH2)
E_H1 = np.sqrt(errorH1) / np.sqrt(semiH2)

# Ajustements séparés en log-log:
# log(E_L2) ~ log(C_L2) + (k+1) log(h)
# log(E_H1) ~ log(C_H1) + k log(h)
x  = np.log(h)
yL = np.log(E_L2)
yH = np.log(E_H1)

pL2, aL2 = np.polyfit(x, yL, 1)  # pente ~ k+1
pH1, aH1 = np.polyfit(x, yH, 1)  # pente ~ k
k_from_L2 = pL2 - 1.0
k_from_H1 = pH1
C_L2 = np.exp(aL2)
C_H1 = np.exp(aH1)


print("\n=== Estimations séparées ===")
print(f"k (depuis L2) ≈ {k_from_L2:.4f}  => L2 ~ h^{{k+1}} avec pente {pL2:.4f}")
print(f"k (depuis H1) ≈ {k_from_H1:.4f}  => H1 ~ h^{{k}}   avec pente {pH1:.4f}")
print(f"C_L2 ≈ {C_L2:.4e},  C_H1 ≈ {C_H1:.4e}")


# Estimation "commune": k partagé, constantes séparées
k_shared = 0.5*(k_from_L2 + k_from_H1)
aL2_shared = np.mean(yL - (k_shared+1.0)*x)
aH1_shared = np.mean(yH -  k_shared*x)
C_L2_shared = np.exp(aL2_shared)
C_H1_shared = np.exp(aH1_shared)

print("\n=== Estimation commune (k partagé) ===")
print(f"k ≈ {k_shared:.4f}")
print(f"C_L2 ≈ {C_L2_shared:.4e},  C_H1 ≈ {C_H1_shared:.4e}")

# Détection d'un M (stabilisation de la pente)
def slope_stability(x, y, min_pts=4, tol=0.05):
    """
    Renvoie M (indice de départ du régime asymptotique)
    et la pente estimée, en cherchant le plus petit M tel que
    la pente entre [M:] et [M+1:] varie de moins de 'tol'.
    """
    n = len(x)
    for m in range(0, n-min_pts+1):
        p, _ = np.polyfit(x[m:], y[m:], 1)
        if m < n-min_pts:
            p_next, _ = np.polyfit(x[m+1:], y[m+1:], 1)
            if abs(p - p_next) < tol:
                return m, p
    # par défaut, tout le jeu
    return 0, np.polyfit(x, y, 1)[0]

M_L2, p_asym_L2 = slope_stability(x, yL)
M_H1, p_asym_H1 = slope_stability(x, yH)
print("\n=== Détection du régime asymptotique ===")
print(f"M_L2 ≈ {M_L2} (pente ~ {p_asym_L2:.4f}),   M_H1 ≈ {M_H1} (pente ~ {p_asym_H1:.4f})")

# ---------------------------
# TRACÉS
# ---------------------------

# (1) Convergence spatiale (log-log) + modèles
plt.figure(figsize=(7,5))
plt.loglog(h, E_L2, 'o-', label=r'$E_{L^2}=\|u-u_h\|_{0,2}/|u|_{2,2}$')
plt.loglog(h, E_H1, 's-', label=r'$E_{H^1}=\|u-u_h\|_{1,2}/|u|_{2,2}$')
plt.loglog(h, C_L2_shared*h**(k_shared+1.0), '--', label=r'$C\,h^{k+1}$ (fit)')
plt.loglog(h, C_H1_shared*h**(k_shared),     '--', label=r'$C\,h^{k}$ (fit)')
plt.xlabel('h')
plt.ylabel('erreur normalisée')
plt.title('Convergence spatiale & superposition des modèles')
plt.legend()
plt.grid(True, which='both', ls=':')

# (2) Pentes locales (optionnel): on peut approximer la pente entre points successifs
# pour visualiser la stabilisation (commenté par défaut)
slopes_L2 = np.diff(np.log(E_L2)) / np.diff(np.log(h))
slopes_H1 = np.diff(np.log(E_H1)) / np.diff(np.log(h))
plt.figure(figsize=(7,5))
plt.plot(0.5*(np.log(h[:-1])+np.log(h[1:])), slopes_L2, 'o-', label='pente L2 (~k+1)')
plt.plot(0.5*(np.log(h[:-1])+np.log(h[1:])), slopes_H1, 's-', label='pente H1 (~k)')
plt.axhline(k_shared+1.0, ls='--')
plt.axhline(k_shared, ls='--')
plt.xlabel('log(h) (milieu)')
plt.ylabel('pente locale')
plt.legend()
plt.title('Pentes locales (diagnostic)')
plt.grid(True, ls=':')

# ---------------------------
# EXPORT DES DONNÉES
# ---------------------------
# Sauvegarde CSV des mesures (utile pour rapport)
data = np.column_stack([
    h,
    np.sqrt(errorL2), np.sqrt(errorH1), np.sqrt(semiH2),  # normes
    E_L2, E_H1                                         # erreurs normalisées
])
header = "h,||u-uh||_L2,||u-uh||_H1,|u|_H2,E_L2_norm,E_H1_norm"
np.savetxt("adrs_convergence_data.csv", data, delimiter=",", header=header, comments="")
print('\nDonnées sauvegardées dans "adrs_convergence_data.csv".')

plt.show()
