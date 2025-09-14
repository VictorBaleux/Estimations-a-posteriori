import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Paramètres du modèle (à ajuster)
# ==========================================================
L      = 1.0        # longueur du domaine [0, L]
v      = 0.750        # vitesse de convection (constante)
nu     = 1e-2       # diffusivité
lam    = 0.0        # lambda de réaction (>=0 typiquement)
T      = 1.0        # temps final
N      = 500        # N intervalles => N+1 points de grille
dt     = 5e-4       # pas de temps

# Fonctions sources et CL (modifiez librement)
def f(t, x):
    # source volumique (exemple: nulle)
    return 0.0

def ul(t):
    # Dirichlet à gauche (exemple: constant et non nul)
    return 0.0

def g(t):
    # Neumann à droite u_x(t,L)=g(t) (exemple: constant et non nul)
    return 0.0

# Paramètres de la gaussienne de départ (utilisée dans u0)
xc     = 0.2 * L      # centre
sigma  = 0.05 * L     # largeur
A_amp  = 1.0          # amplitude libre de la gaussienne
# ==========================================================


# ==========================================================
# Construction de u0(x) compatible avec u(0)=ul(0) et u_x(L)=g(0)
#    u0(x) = A*G(x) + B*x + C
#    G(x) = exp(-(x-xc)^2/(2*sigma^2))
# Conditions:
#    u0(0)=ul(0)  => C = ul(0) - A*G(0)
#    u0'(L)=g(0)  => u0'(x)=A*(-(x-xc)/sigma^2)G(x) + B
#                    B = g(0) - A*(-(L-xc)/sigma^2)*G(L)
# ==========================================================
def build_u0(x, A=A_amp, xc=xc, sigma=sigma):
    G0 = np.exp(-((x - xc)**2) / (2.0 * sigma**2))
    G0_at_0 = np.exp(-((0.0 - xc)**2) / (2.0 * sigma**2))
    G0_at_L = np.exp(-((L   - xc)**2) / (2.0 * sigma**2))

    B = g(0.0) - A * (-(L - xc) / sigma**2) * G0_at_L
    C = ul(0.0) - A * G0_at_0

    u0 = A * G0 + B * x + C
    return u0


# ==========================================================
# Assemblage du système linéaire implicite:
# (I - dt * L) u^{n+1} = u^n + dt f^{n+1}
# avec L = -v D1_upwind + nu D2 - lam I
# On encode directement A = I + dt*v*D1_upwind - dt*nu*D2 + dt*lam*I
# et on remplace les deux lignes de bord par les CL (Dirichlet/Neumann).
# ==========================================================
def build_matrix(N, L, dt, v, nu, lam):
    dx = L / N
    size = N + 1
    A = np.zeros((size, size))

    # Upwind pour l'advection (au niveau des points i=1..N-1)
    # D1_upwind u_i ~ (u_i - u_{i-1})/dx si v>0
    #              ~ (u_{i+1} - u_i)/dx si v<0
    use_backward = (v >= 0.0)

    # Remplissage lignes intérieures i=1..N-1
    for i in range(1, N):
        # Diffusion (D2 central): u_{i-1} - 2 u_i + u_{i+1} sur dx^2
        A[i, i]   += 1.0 + dt * lam + dt * nu * (2.0 / dx**2)
        A[i, i-1] += - dt * nu * (1.0 / dx**2)
        A[i, i+1] += - dt * nu * (1.0 / dx**2)

        # Advection implicite upwind
        if use_backward:
            # D1 ≈ (u_i - u_{i-1})/dx
            A[i, i]   += dt * v * ( 1.0 / dx)
            A[i, i-1] += dt * v * (-1.0 / dx)
        else:
            # D1 ≈ (u_{i+1} - u_i)/dx
            A[i, i]   += dt * v * (-1.0 / dx)
            A[i, i+1] += dt * v * ( 1.0 / dx)

    # Bord gauche (Dirichlet) : u(t,0) = ul(t)
    A[0, :] = 0.0
    A[0, 0] = 1.0

    # Bord droit (Neumann) : (u_N - u_{N-1})/dx = g(t)
    A[N, :] = 0.0
    A[N, N-1] = -1.0 / dx
    A[N, N]   =  1.0 / dx

    return A


def step_rhs(u_prev, t_next, x, dt):
    # RHS b = u^n + dt f(t^{n+1}, x) pour points intérieurs
    b = u_prev.copy()
    b[1:-1] += dt * np.array([f(t_next, xi) for xi in x[1:-1]])

    # CL de Dirichlet au bord gauche
    b[0] = ul(t_next)

    # CL de Neumann au bord droit
    b[-1] = g(t_next)

    return b


def solve_pde(L, v, nu, lam, T, N, dt):
    x = np.linspace(0.0, L, N + 1)
    A = build_matrix(N, L, dt, v, nu, lam)

    # Pré-calcul: factorisation possible (si SciPy dispo), ici solve dense simple
    # Création de u^0 compatible
    u = build_u0(x)

    # Petits checks de compatibilité (informative, tolérance ~1e-8)
    dx = L / N
    dL_num = (u[-1] - u[-2]) / dx
    if abs(u[0] - ul(0.0)) > 1e-8 or abs(dL_num - g(0.0)) > 1e-6:
        print("[Avertissement] u0 n'est pas parfaitement compatible numériquement avec les CL.")

    times_to_store = np.linspace(0.0, T, 5)  # 5 instants (dont t=0 et t=T)
    snapshots = [(0.0, u.copy())]
    t = 0.0
    next_store_idx = 1  # le prochain instant à mémoriser (sauter t=0 déjà stocké)

    while t < T - 1e-12:
        t_next = min(t + dt, T)
        b = step_rhs(u, t_next, x, dt)
        u = np.linalg.solve(A, b)
        t = t_next

        # stocker si on a franchi le prochain jalon
        while next_store_idx < len(times_to_store) and t >= times_to_store[next_store_idx] - 1e-12:
            snapshots.append((times_to_store[next_store_idx], u.copy()))
            next_store_idx += 1

    return x, snapshots


# ==========================================================
# Lancement + visualisation
# ==========================================================
x, snapshots = solve_pde(L, v, nu, lam, T, N, dt)

plt.figure()
for (ti, ui) in snapshots:
    plt.plot(x, ui, label=f"t={ti:.3f}")
plt.xlabel("x")
plt.ylabel("u(t,x)")
plt.title("Convection–Diffusion–Réaction 1D (Dirichlet gauche, Neumann droite)")
plt.legend()
plt.tight_layout()
plt.savefig("Convection_Diffusion_Réaction_1D.png", dpi=300, bbox_inches="tight")
plt.show()