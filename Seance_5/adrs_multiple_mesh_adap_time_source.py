# -*- coding: utf-8 -*-
"""
adrs_multiple_mesh_adap_time_source.py (clean plots)


"""
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------- Paramètres physiques -----------------------
K = 0.01        # Diffusion
V = 1.0         # Advection
lamda = 1.0     # Réaction
xmin, xmax = 0.0, 1.0
Time = 1.0      # On s'arrête à 1s pour les tracés demandés

# ----------------------- Paramètres numériques ----------------------
NX_init = 5             # Points initiaux pour lancer l'adaptation
NT_max = 200000         # Garde-fou sur le nombre de pas de temps
plot_every = 10**9      # Désactivé
# Schéma : Euler explicite

# ----------------------- Paramètres adaptation ----------------------
hmin, hmax = 0.005, 0.15
err_curv = 0.01298        # seuil pour la métrique locale |u_xx|/err_curv
niter_refinement_max = 10

# Critère d'arrêt MIXTE (ne pas arrêter tant que les 2 ne sont pas atteints)
NX_min_required = 80    # nombre minimal de points de maillage
L2_tol = 1e-3           # tolérance sur l'erreur L2 à t=Time

# Option d'utilisation de la métrique en moyenne temporelle (True) ou stationnaire finale (False)
USE_TIME_AVG_METRIC = True

# Maillage de fond pour interpoler et accumuler la métrique en temps
NX_background = 400
background_mesh = np.linspace(xmin, xmax, NX_background)

# ----------------------- Fonctions utilitaires ----------------------
def u_time(t):
    """u(t) = sin(4*pi*t)"""
    return math.sin(4.0*math.pi*t)

def du_time(t):
    """u'(t) = 4*pi*cos(4*pi*t)"""
    return 4.0*math.pi*math.cos(4.0*math.pi*t)

def v_profile(x):
    """v(x) = somme de gaussiennes (même que Tex de la version initiale)"""
    return 2.0*np.exp(-100.0*(x-(xmax+xmin)*0.25)**2) + np.exp(-200.0*(x-(xmax+xmin)*0.65)**2)

def central_first_derivative_nonuniform(x, y):
    """Dérivée première sur maillage non uniforme (ordre 1 centré)."""
    n = len(x)
    yp = np.zeros_like(y)
    for j in range(1, n-1):
        yp[j] = (y[j+1]-y[j-1])/(x[j+1]-x[j-1])
    yp[0] = yp[1]
    yp[-1] = yp[-2]
    return yp

def central_second_derivative_nonuniform(x, y):
    """Dérivée seconde sur maillage non uniforme à partir de pentes centrées."""
    n = len(x)
    yx = central_first_derivative_nonuniform(x, y)
    yxx = np.zeros_like(y)
    for j in range(1, n-1):
        yx_ip1 = (y[j+1]-y[j])/(x[j+1]-x[j])
        yx_im1 = (y[j]-y[j-1])/(x[j]-x[j-1])
        denom = 0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1])
        yxx[j] = (yx_ip1 - yx_im1)/denom
    yxx[0] = yxx[1]
    yxx[-1] = yxx[-2]
    return yxx

def interpolate_piecewise_linear(x_src, y_src, x_query):
    """Interpolation linéaire 1D (x_src croissant). Bords étendus par valeurs aux bords."""
    yq = np.empty_like(x_query)
    i = 0
    for k, xq in enumerate(x_query):
        if xq <= x_src[0]:
            yq[k] = y_src[0]; continue
        if xq >= x_src[-1]:
            yq[k] = y_src[-1]; continue
        while not (x_src[i] <= xq <= x_src[i+1]):
            i += 1
        t = (xq - x_src[i])/(x_src[i+1] - x_src[i])
        yq[k] = (1.0-t)*y_src[i] + t*y_src[i+1]
    return yq

def build_new_mesh_from_hloc(x_old, hloc, hmin, hmax):
    """Re-construit un nouveau maillage en suivant les longueurs locales désirées hloc."""
    xnew = [xmin]
    while xnew[-1] < xmax - hmin:
        for i in range(len(x_old)-1):
            if x_old[i] <= xnew[-1] <= x_old[i+1]:
                h_here = (hloc[i]*(x_old[i+1]-xnew[-1]) + hloc[i+1]*(xnew[-1]-x_old[i]))/(x_old[i+1]-x_old[i])
                h_here = min(max(hmin, h_here), hmax)
                xnext = min(xmax, xnew[-1] + h_here)
                xnew.append(xnext)
                break
    return np.array(xnew)

if __name__ == "__main__":
    # Nettoyage de toutes les figures au démarrage
    plt.close("all")

    itera = 0
    NX = NX_init
    errorL2_hist = []
    NX_hist = []

    last_hloc_stationary = None
    last_hloc_timeavg = None

    # Pour les tracés finaux (évite les doublons)
    times_to_save = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    snapshots_final = {}
    residuals_final = []
    times_r_final = []
    Tex_final = None
    x_final = None

    while True:
        itera += 1
        x = np.linspace(xmin, xmax, NX)
        T = np.zeros_like(x)  # u(0)=0 => T(x,0)=0

        v = v_profile(x)
        vx = central_first_derivative_nonuniform(x, v)
        vxx = central_second_derivative_nonuniform(x, v)

        F_spatial = V*vx - K*vxx + lamda*v

        dx_local = np.diff(x)
        dx_min = np.min(dx_local)
        dt_diff = 0.45 * dx_min*dx_min / (K + 1e-14)
        dt_adv = 0.45 * dx_min / (abs(V) + 1e-14)
        dt = min(dt_diff, dt_adv)
        if dt <= 0:
            dt = 1e-4

        t = 0.0
        nstep = 0
        residuals = []
        times_r = []
        snapshots = {}
        Mback_sum = np.zeros_like(background_mesh)
        Mback_count = 0
        to_save_set = set(times_to_save)

        while t < Time and nstep < NT_max:
            nstep += 1
            if t + dt > Time:
                dt = Time - t

            Tx = central_first_derivative_nonuniform(x, T)
            Txx = central_second_derivative_nonuniform(x, T)

            visnum = np.zeros_like(T)
            for j in range(1, len(x)-1):
                visnum[j] = 0.5*(0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))*abs(V)
            xnu = K + visnum

            ut = u_time(t)
            dut = du_time(t)
            F_time = dut*v + ut*F_spatial

            RHS = np.zeros_like(T)
            for j in range(1, len(x)-1):
                RHS[j] = dt * (-V*Tx[j] + xnu[j]*Txx[j] - lamda*T[j] + F_time[j])

            T[1:-1] += RHS[1:-1]
            T[-1] = T[-2]

            res = float(np.sum(np.abs(RHS[1:-1])))
            residuals.append(res)
            times_r.append(t+dt)

            for tk in sorted(list(to_save_set)):
                if t < tk <= t+dt + 1e-14:
                    snapshots[tk] = (x.copy(), T.copy())
                    to_save_set.remove(tk)

            metric_nodes = np.minimum(1.0/hmin**2, np.maximum(1.0/hmax**2, np.abs(Txx)/err_curv))
            Mback_sum += interpolate_piecewise_linear(x, metric_nodes, background_mesh)
            Mback_count += 1

            t += dt

        uT = u_time(Time)
        Tex = uT * v

        # Erreur L2 finale
        errL2 = 0.0
        for j in range(1, len(x)-1):
            wj = 0.5*(x[j+1]-x[j-1])
            errL2 += wj * (T[j]-Tex[j])**2
        errL2 = math.sqrt(max(errL2, 0.0))

        errorL2_hist.append(errL2)
        NX_hist.append(NX)

        Txx_final = central_second_derivative_nonuniform(x, T)
        metric_stationary = np.minimum(1.0/hmin**2, np.maximum(1.0/hmax**2, np.abs(Txx_final)/err_curv))
        hloc_stationary = 1.0/np.sqrt(metric_stationary)
        last_hloc_stationary = (x.copy(), hloc_stationary.copy())

        Mback_avg = Mback_sum / max(Mback_count, 1)
        metric_timeavg_nodes = interpolate_piecewise_linear(background_mesh, Mback_avg, x)
        metric_timeavg_nodes = np.minimum(1.0/hmin**2, np.maximum(1.0/hmax**2, metric_timeavg_nodes))
        hloc_timeavg = 1.0/np.sqrt(metric_timeavg_nodes)
        last_hloc_timeavg = (x.copy(), hloc_timeavg.copy())

        # Conserve UNIQUEMENT les données de cette itération (dernière si on sort)
        snapshots_final = snapshots
        residuals_final = residuals
        times_r_final = times_r
        Tex_final = Tex
        x_final = x

        # Critère d'arrêt mixte
        hloc_to_use = hloc_timeavg if USE_TIME_AVG_METRIC else hloc_stationary
        x_new = build_new_mesh_from_hloc(x, hloc_to_use, hmin, hmax)
        NX_new = len(x_new)

        cond_points = (NX_new >= NX_min_required)
        cond_error = (errL2 <= L2_tol)
        print(f"[Iter {itera}] NX_old={NX}, NX_new={NX_new}, L2={errL2:.3e}, "
              f"points_ok={cond_points}, error_ok={cond_error}")

        if (cond_points and cond_error) or itera >= niter_refinement_max:
            break
        NX = NX_new

    # ----------------------- TRACÉS (une seule fois) ---------------------
    # 1) Solution u(x,t) aux instants demandés
    plt.figure("Solution u(x,t) à différents instants"); plt.clf()
    for tk in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if tk in snapshots_final:
            xs, Ts = snapshots_final[tk]
            plt.plot(xs, Ts, label=f"t={tk:.1f}s")
    if Tex_final is not None:
        plt.plot(x_final, Tex_final, label="u_ex(x,1.0s)")
    plt.xlabel("x"); plt.ylabel("u"); plt.legend()
    plt.title("u(x,t) pour t ∈ {0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0}")

    # 2) Résidu instationnaire
    plt.figure("Résidu vs temps (instationnaire)"); plt.clf()
    if len(times_r_final)>0:
        plt.plot(np.array(times_r_final), np.array(residuals_final))
    plt.xlabel("temps"); plt.ylabel("résidu (somme |RHS|)")
    plt.title("Le résidu ne converge pas vers 0 (forçage périodique)")

    # 3) h(x) final : stationnaire vs moyenne en temps (deux courbes sans doublons)
    plt.figure("Distribution h(x) finale"); plt.clf()
    xs, hs = last_hloc_stationary
    plt.plot(xs, hs, label="h_stationnaire(t=Time)")
    xt, ht = last_hloc_timeavg
    plt.plot(xt, ht, label="h_moyenne_temps")
    plt.xlabel("x"); plt.ylabel("h local"); plt.legend()
    plt.title("Longueurs locales désirées h(x)")

    iters = np.arange(1, len(errorL2_hist)+1)

    # Sauvegardes
    plt.figure("Solution u(x,t) à différents instants")
    plt.savefig("solutions_instants_os.png", dpi=150, bbox_inches="tight")
    plt.figure("Résidu vs temps (instationnaire)")
    plt.savefig("residu_vs_temps_os.png", dpi=150, bbox_inches="tight")
    plt.figure("Distribution h(x) finale")
    plt.savefig("h_distribution_os.png", dpi=150, bbox_inches="tight")
   
