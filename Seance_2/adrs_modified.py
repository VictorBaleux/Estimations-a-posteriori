
import numpy as np
import math
import matplotlib.pyplot as plt

# ================================================================
# ADRS 1D (Advection-Diffusion-Reaction + Source) — version modifiée
# ================================================================
# Equation (stationnaire via marche en temps) :
#   u_t + v u_s - nu u_ss + lambda u = f(s)
#
# Objectif du TP :
# - Utiliser la solution exacte u_ex(s) = exp(-10*(s-0.5)^2)
# - Construire le terme source f(s) = v u_s - nu u_ss + lambda u
# - Identifier dans le code :
#   (i)  "décentrage = centrage + viscosité numérique" (chap 12)
#   (ii) condition de stabilité CFL (chap 10)
#   (iii) marche en temps vers la solution stationnaire
#   (iv) condition en sortie actuellement et implémentation de u_s(L)=0
# - Vérifier la convergence vers la solution stationnaire pour NX=100
# - Tracer la convergence ||u^{n+1}-u^n||_L2 normalisée
# - Calculer après convergence les normes L2 et H1 pour 5 maillages (en partant de NX=3)
# - Tracer erreurs L2 et H1 en fonction de h=dx (et sauvegarder les figures)
#
# Remarque : on impose au bord gauche une condition de Dirichlet exacte u(0)=u_ex(0),
#            et au bord droit la condition de Neumann u_s(L)=0 (sortie).
# ================================================================

# --------------------
# Paramètres physiques
# --------------------
L = 1.0
v = 1.0
nu = 0.01
lam = 1.0

# ----------------------
# Solution exacte & f(s)
# ----------------------
def u_ex(s):
    # u_ex(s) = exp(-10*(s-0.5)**2)
    return np.exp(-10.0*(s-0.5)**2)

def u_ex_s(s):
    # u_s(s) = -20*(s-0.5)*u_ex(s)
    ue = u_ex(s)
    return -20.0*(s-0.5)*ue

def u_ex_ss(s):
    # u_ss(s) = -20*u_ex + 400*(s-0.5)**2 * u_ex
    ue = u_ex(s)
    return (-20.0 + 400.0*(s-0.5)**2)*ue

def f_source(s):
    # f(s) = v u_s - nu u_ss + lam u
    return v*u_ex_s(s) - nu*u_ex_ss(s) + lam*u_ex(s)

# --------------------------------------
# Boucle en temps (marche vers station.) 
# --------------------------------------
def solve_stationary(NX, NT_max=200000, eps=1e-9, plot_each=None, save_prefix="run"):
    """
    Marche en temps explicite jusqu'à stationnaire.
    Renvoie :
      x, T (solution numérique),
      res_hist (suite des ||u^{n+1}-u^n||_L2),
      dt, dx
    """
    x = np.linspace(0.0, L, NX)
    dx = x[1]-x[0]
    T = np.zeros_like(x)  # départ nul
    RHS = np.zeros_like(x)
    res_hist = []

    # ---- (i) "décentrage = centrage + viscosité numérique" (chap 12) ----
    # On ajoute une viscosité numérique proportionnelle à |v| dx /2 :
    # xnu = nu + 0.5*dx*|v|
    # -> Diffusion effective stabilisée pour la partie advective.
    def xnu():
        return nu + 0.5*dx*abs(v)

    # Terme source (fabrique de la solution)
    F = f_source(x)

    # ---- (ii) Condition de stabilité CFL (chap 10) ----
    # Schéma explicite : dt borné par contributions advection + diffusion + réaction + source (sûreté)
    # Formule pratique (conservatrice) :
    dt = dx*dx / (abs(v)*dx + 2.0*xnu() + (abs(np.max(F))+lam)*dx*dx + 1e-30)

    # Pour information :
    # print(f"NX={NX}, dx={dx:.4e}, dt={dt:.4e}")

    # Conditions aux limites :
    #   - Bord gauche (s=0) : Dirichlet exact -> T[0] = u_ex(0)
    #   - Bord droit (s=L) : Neumann (u_s(L)=0) -> T[-1] = T[-2] (discrétisation simple)
    T[0]  = u_ex(x[0])
    T[-1] = T[-2] if NX >= 2 else u_ex(x[-1])

    n = 0
    res0 = None

    while n < NT_max:
        n += 1
        T_old = T.copy()

        # Discrétisation spatiale (centrée + viscosité numérique via xnu) sur points intérieurs
        for j in range(1, NX-1):
            # Gradient et laplacien centrés
            Tx  = (T_old[j+1]-T_old[j-1])/(2.0*dx)
            Txx = (T_old[j-1] - 2.0*T_old[j] + T_old[j+1])/(dx*dx)

            # ---- (i) encore : on utilise xnu() comme "nu effectif" = nu + viscosité numérique ----
            nu_eff = xnu()

            # Mise à jour explicite (Euler) du résidu local
            RHS[j] = dt*(-v*Tx + nu_eff*Txx - lam*T_old[j] + F[j])

        # Update solution
        T[1:-1] = T_old[1:-1] + RHS[1:-1]

        # Conditions aux limites à chaque pas de temps
        # Bord gauche : Dirichlet exact
        T[0] = u_ex(x[0])
        # Bord droit : ---- (iv) Implémentation de u_s(L)=0 ----
        # Neumann : (T_N - T_{N-1})/dx = 0 -> T_N = T_{N-1}
        T[-1] = T[-2]

        # ---- (iii) Marche en temps vers la solution stationnaire ----
        # On surveille la décroissance de ||u^{n+1}-u^n||_L2
        diff = T - T_old
        res = math.sqrt(np.sum(diff*diff)*dx)
        if res0 is None and res > 0:
            res0 = res
        res_hist.append(res/(res0 if res0 else 1.0))

        # Critère d'arrêt
        if res0 and res/res0 < eps:
            break

    return x, T, np.array(res_hist), dt, dx

# ---------------------------------------------
# Fonctions d'erreur L2 et H1 (après convergence)
# ---------------------------------------------
def compute_errors_L2_H1(x, T):
    dx = x[1]-x[0] if len(x) > 1 else 1.0
    ue = u_ex(x)
    ue_s = u_ex_s(x)

    # Erreur L2
    err_L2 = math.sqrt(np.sum((T-ue)**2) * dx)

    # Erreur H1 (semi-norme) : || dT/dx - u_ex_s ||_L2
    dTdx = np.zeros_like(T)
    if len(T) >= 3:
        dTdx[1:-1] = (T[2:] - T[:-2])/(2.0*dx)
        # Bords : une version au premier ordre (peu d'impact sur l'ordre global)
        dTdx[0]  = (T[1]-T[0])/dx
        dTdx[-1] = (T[-1]-T[-2])/dx
    err_H1 = math.sqrt(np.sum((dTdx - ue_s)**2) * dx)

    return err_L2, err_H1

# =====================
# 1) Run NX=100 (plots)
# =====================
NX_convergence = 100
x, T, res_hist, dt, dx = solve_stationary(NX_convergence, NT_max=500000, eps=1e-10, save_prefix="nx100")

# Figures : solution vs exact, et convergence en temps
plt.figure(figsize=(6,4))
plt.plot(x, T, label="u numérique (NX=100)")
plt.plot(x, u_ex(x), '--', label="u exact")
plt.xlabel("s")
plt.ylabel("u")
plt.title("Solution stationnaire (NX=100)")
plt.legend()
plt.tight_layout()
plt.savefig("solution_N100.png", dpi=200)  # -> image enregistrée
plt.show()

plt.figure(figsize=(6,4))
if len(res_hist) > 0 and res_hist[0] != 0:
    plt.plot(np.log10(res_hist), lw=1.5)
    plt.ylabel("log10(||u^{n+1}-u^n|| / ||u^1-u^0||)")
else:
    plt.plot(res_hist, lw=1.5)
    plt.ylabel("||u^{n+1}-u^n|| / ||u^1-u^0||")
plt.xlabel("Itérations temps")
plt.title("Convergence vers l'état stationnaire (NX=100)")
plt.tight_layout()
plt.savefig("convergence_history_N100.png", dpi=200)  # -> image enregistrée
plt.show()

# ============================================
# 2) Erreurs L2/H1 pour 5 maillages (NX >= 3)
#    partant de 3 points
# ============================================
NX_list = [3, 5, 9, 17, 33]  # 5 maillages, début à 3 points
h_list = []
errL2_list = []
errH1_list = []

for NX in NX_list:
    xh, Th, res_h, dt_h, dx_h = solve_stationary(NX, NT_max=400000, eps=1e-12)
    eL2, eH1 = compute_errors_L2_H1(xh, Th)
    h_list.append(dx_h)
    errL2_list.append(eL2)
    errH1_list.append(eH1)

# Tracé des erreurs sur un même graphe
plt.figure(figsize=(6,4))
plt.loglog(h_list, errL2_list, 'o-', label='Erreur L2')
plt.loglog(h_list, errH1_list, 's-', label='Erreur H1')
plt.gca().invert_xaxis()
plt.xlabel("h = dx")
plt.ylabel("Erreur")
plt.title("Erreurs vs h (5 maillages)")
plt.legend()
plt.tight_layout()
plt.savefig("error_vs_h.png", dpi=200)  # -> image enregistrée
plt.show()

print("Terminé. Images sauvegardées :")
print("- solution_N100.png")
print("- convergence_history_N100.png")
print("- error_vs_h.png")
