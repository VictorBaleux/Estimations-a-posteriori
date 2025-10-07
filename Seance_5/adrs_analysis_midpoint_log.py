
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------------------
# Paramètres du problème
# ----------------------------
@dataclass
class Params:
    V: float = 1.0   # vitesse de convection
    K: float = 0.1   # diffusion
    lam: float = 1.0 # réaction
    L: float = 1.0   # longueur du domaine [0,L]
    Time: float = 1.0  # temps final

# Solution exacte utilisée pour construire f
def exact_u(x, t):
    L = 1.0
    v_env = np.exp(-1000*((x - L/3.0)/L)**2)
    v = (v_env + np.exp(-10*v_env)) * np.sin(5*np.pi*x/L)
    return np.sin(4*np.pi*t) * v

def exact_ut(x, t):
    L = 1.0
    v_env = np.exp(-1000*((x - L/3.0)/L)**2)
    v = (v_env + np.exp(-10*v_env)) * np.sin(5*np.pi*x/L)
    return 4*np.pi*np.cos(4*np.pi*t) * v

# Dérivées spatiales centrées
def centered_first_derivative(u, dx):
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    return ux

def centered_second_derivative(u, dx):
    uxx = np.zeros_like(u)
    uxx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / (dx*dx)
    return uxx

# Terme source
def forcing_f(x, t, p: Params):
    u = exact_u(x, t)
    ut = exact_ut(x, t)
    dx = x[1]-x[0]
    ux = centered_first_derivative(u, dx)
    uxx = centered_second_derivative(u, dx)
    f = ut + p.V*ux - p.K*uxx + p.lam*u
    f[0] = 0.0
    f[-1] = 0.0
    return f

# Opérateur spatial
def spatial_operator(T, x, t, p: Params):
    dx = x[1]-x[0]
    Tx = centered_first_derivative(T, dx)
    Txx = centered_second_derivative(T, dx)
    rhs = -p.V*Tx + p.K*Txx - p.lam*T + forcing_f(x, t, p)
    rhs[0] = 0.0
    rhs[-1] = 0.0
    return rhs

# Condition CFL pour le pas de temps
def cfl_dt(x, p: Params, safety=0.9):
    dx = x[1]-x[0]
    choices = []
    if p.V != 0:
        choices.append(dx/abs(p.V))
    if p.K > 0:
        choices.append(dx*dx/(2*p.K))
    choices.append(1.0/max(p.lam, 1e-12))
    return safety * min(choices)

# Runge-Kutta d'ordre 1 à 4
def step_rk(T, t, dt, x, p: Params, order=4):
    if order == 1:
        k1 = spatial_operator(T, x, t, p)
        return T + dt * k1
    elif order == 2:
        k1 = spatial_operator(T, x, t, p)
        k2 = spatial_operator(T + 0.5*dt*k1, x, t + 0.5*dt, p)
        return T + dt * k2
    elif order == 3:
        k1 = spatial_operator(T, x, t, p)
        T1 = T + dt*k1
        k2 = spatial_operator(T1, x, t + dt, p)
        T2 = 0.75*T + 0.25*(T1 + dt*k2)
        k3 = spatial_operator(T2, x, t + 0.5*dt, p)
        return (1.0/3.0)*T + (2.0/3.0)*(T2 + dt*k3)
    elif order == 4:
        k1 = spatial_operator(T, x, t, p)
        k2 = spatial_operator(T + 0.5*dt*k1, x, t + 0.5*dt, p)
        k3 = spatial_operator(T + 0.5*dt*k2, x, t + 0.5*dt, p)
        k4 = spatial_operator(T + dt*k3, x, t + dt, p)
        return T + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    else:
        raise ValueError("order must be 1, 2, 3, or 4.")

# Erreur L2
def l2_error(T, Tex, dx):
    return math.sqrt(np.sum((T - Tex)**2) * dx)

# ------------------------------------------------------
# 1) Erreur à T/2 et T pour différents maillages
# ------------------------------------------------------
def run_convergence(order=4):
    mesh_list = list(range(10, 101, 10))
    p = Params()
    err_half, err_final, h_list = [], [], []
    for NX in mesh_list:
        x = np.linspace(0.0, p.L, NX)
        h = x[1]-x[0]
        T = np.zeros_like(x)
        t = 0.0
        dt = cfl_dt(x, p, safety=0.8)
        half_time = 0.5 * p.Time
        half_recorded = False
        while t < p.Time - 1e-14:
            if t + dt > p.Time:
                dt = p.Time - t
            T = step_rk(T, t, dt, x, p, order=order)
            t += dt
            if (not half_recorded) and t >= half_time:
                Tex_half = exact_u(x, half_time)
                err_half.append(l2_error(T, Tex_half, h))
                half_recorded = True
        Tex_T = exact_u(x, p.Time)
        err_final.append(l2_error(T, Tex_T, h))
        h_list.append(h)
    return np.array(h_list), np.array(err_half), np.array(err_final)

def plot_convergence(order=4):
    h, E_half, E_T = run_convergence(order=order)
    coef_half = np.polyfit(np.log(h), np.log(E_half + 1e-30), 1)
    coef_T = np.polyfit(np.log(E_T + 1e-30), np.log(E_T + 1e-30), 1)  # keep same behavior as before
    # Fix potential mistake: slope should be computed vs log(h)
    coef_T = np.polyfit(np.log(h), np.log(E_T + 1e-30), 1)
    p_half, p_T = coef_half[0], coef_T[0]
    print(f"[Convergence] RK{order}: slope at T/2 = {p_half:.3f}")
    print(f"[Convergence] RK{order}: slope at T   = {p_T:.3f}")

    plt.figure()
    plt.loglog(h, E_half, 'o-', label=f"T/2 (pente ~ {p_half:.2f})")
    plt.loglog(h, E_T, 's-', label=f"T (pente ~ {p_T:.2f})")
    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("Erreur L2")
    plt.title(f"Erreur vs h à T/2 et T (RK{order})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 2) Évolution de l’erreur au point milieu (RK1..4) en échelle log
# ------------------------------------------------------
def midpoint_error_vs_time(NX=201, orders=(1,2,3,4)):
    p = Params()
    x = np.linspace(0.0, p.L, NX)
    dx = x[1] - x[0]
    i_mid = int(round(0.5 * (NX-1)))
    x_mid = x[i_mid]
    results = {}
    for order in orders:
        T = np.zeros_like(x)
        t = 0.0
        dt = cfl_dt(x, p, safety=0.8)
        times = [t]
        errs = [abs(T[i_mid] - exact_u(x_mid, t))]
        while t < p.Time - 1e-14:
            if t + dt > p.Time:
                dt = p.Time - t
            T = step_rk(T, t, dt, x, p, order=order)
            t += dt
            times.append(t)
            errs.append(abs(T[i_mid] - exact_u(x_mid, t)))
        results[order] = (np.array(times), np.array(errs))
    return x_mid, results

def plot_midpoint_evolution(NX=201, orders=(1,2,3,4), eps=1e-16):
    """
    Trace |erreur| au point x_mid en fonction du temps pour RK1..4
    en échelle logarithmique (axe Y). On ajoute un petit epsilon pour
    éviter log(0) lorsque l'erreur est nulle (par ex. à t=0).
    """
    x_mid, results = midpoint_error_vs_time(NX=NX, orders=orders)
    plt.figure()
    for order in orders:
        t, e = results[order]
        plt.plot(t, e + eps, label=f"RK{order}")
    plt.yscale('log')
    plt.xlabel("Temps")
    plt.ylabel(f"|erreur| (log) en x={x_mid:.3f}")
    plt.title("Erreur au milieu du domaine vs temps (RK1..RK4)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_convergence(order=4)
    plot_midpoint_evolution(NX=201, orders=(1,2,3,4))
