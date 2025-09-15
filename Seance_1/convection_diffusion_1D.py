import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Paramètres du modèle (à ajuster)
# ==========================================================
L      = 1.0       # longueur du domaine [0, L]
v      = 0.750     # vitesse de convection (constante)
nu     = 1e-2      # diffusivité
lam    = 0.0       # lambda de réaction (>=0 typiquement)
T      = 1.0       # temps final
N      = 500       # N intervalles => N+1 points de grille
dt     = 5e-4      # pas de temps

# Choix du solveur: "sparse" (par défaut) ou "banded"
SOLVER = "sparse"   # "sparse" | "banded"

# Fonctions sources et CL (modifiez librement)
def f(t, x):
    return 0.0

def ul(t):
    return 0.0

def g(t):
    return 0.0

# Paramètres de la gaussienne de départ
xc     = 0.2 * L
sigma  = 0.05 * L
A_amp  = 1.0
# ==========================================================


# ==========================================================
# u0(x) compatible avec u(0)=ul(0) et u_x(L)=g(0)
# ==========================================================
def build_u0(x, A=A_amp, xc=xc, sigma=sigma):
    G0 = np.exp(-((x - xc)**2) / (2.0 * sigma**2))
    G0_at_0 = np.exp(-((0.0 - xc)**2) / (2.0 * sigma**2))
    G0_at_L = np.exp(-((L   - xc)**2) / (2.0 * sigma**2))

    B = g(0.0) - A * (-(L - xc) / sigma**2) * G0_at_L
    C = ul(0.0) - A * G0_at_0
    return A * G0 + B * x + C


# ==========================================================
# Assemblage tri-diagonal (trois diagonales seulement)
# (I - dt*L) u^{n+1} = u^n + dt f^{n+1}
# L = -v D1_upwind + nu D2 - lam I
# => A = I + dt*v*D1_upwind - dt*nu*D2 + dt*lam*I
# CL: Dirichlet à gauche, Neumann à droite.
# ==========================================================
def build_tridiag(N, L, dt, v, nu, lam):
    dx = L / N
    n  = N + 1

    main  = np.zeros(n)     # diagonale principale A[i,i]
    lower = np.zeros(n - 1) # sous-diagonale      A[i, i-1]
    upper = np.zeros(n - 1) # sur-diagonale       A[i, i+1]

    use_backward = (v >= 0.0)
    alpha = dt * nu / dx**2
    beta  = dt * v  / dx

    # Lignes intérieures i=1..N-1
    for i in range(1, N):
        # Diffusion centrale
        main[i]  += 1.0 + dt*lam + 2.0 * alpha
        lower[i-1] += -alpha
        upper[i]   += -alpha

        # Advection upwind implicite
        if use_backward:
            main[i]  +=  beta
            lower[i-1] += -beta
        else:
            main[i]  += -beta
            upper[i] +=  beta

    # Bord gauche (Dirichlet): u(.,0) = ul
    main[0] = 1.0
    # upper[0] reste 0, lower[0] n'est pas utilisé

    # Bord droit (Neumann): (u_N - u_{N-1})/dx = g
    main[-1]  =  1.0 / dx
    lower[-1] = -1.0 / dx
    # upper[-1] inexistant pour la dernière ligne

    return lower, main, upper


def step_rhs(u_prev, t_next, x, dt):
    b = u_prev.copy()
    b[1:-1] += dt * np.array([f(t_next, xi) for xi in x[1:-1]])
    # CL
    b[0]  = ul(t_next)  # Dirichlet
    b[-1] = g(t_next)   # Neumann (cohérent avec la dernière ligne)
    return b


def solve_pde(L, v, nu, lam, T, N, dt, solver="sparse"):
    x = np.linspace(0.0, L, N + 1)

    # Assemblage tri-diagonal
    lower, main, upper = build_tridiag(N, L, dt, v, nu, lam)
    n = N + 1

    # Préparation du solveur choisi (factorisation réutilisée si possible)
    if solver == "sparse":
        # Matrice creuse CSC + LU sparse réutilisable
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        
        A_csc = diags([lower, main, upper], offsets=[-1, 0, 1],
                      shape=(n, n), format="csc")
        lu = splu(A_csc)        # factorisation unique
        def solve_A(rhs):
            return lu.solve(rhs)

    elif solver == "banded":
        # Stockage bande (l=u=1) pour solve_banded
        # ab[0,1:] = upper ; ab[1,:] = main ; ab[2,:-1] = lower
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n), dtype=float)
        ab[0, 1:] = upper
        ab[1, :]  = main
        ab[2, :-1]= lower
        def solve_A(rhs):
            # Remarque: solve_banded refactorise à chaque appel
            return solve_banded((1, 1), ab, rhs, overwrite_ab=False, overwrite_b=False, check_finite=False)
    else:
        raise ValueError("solver doit valoir 'sparse' ou 'banded'.")

    # État initial compatible
    u = build_u0(x)

    # Petits checks CL
    dx = L / N
    dL_num = (u[-1] - u[-2]) / dx
    if abs(u[0] - ul(0.0)) > 1e-8 or abs(dL_num - g(0.0)) > 1e-6:
        print("[Avertissement] u0 n'est pas parfaitement compatible numériquement avec les CL.")

    # Intégration en temps
    times_to_store = np.linspace(0.0, T, 5)
    snapshots = [(0.0, u.copy())]
    t = 0.0
    next_store_idx = 1

    while t < T - 1e-12:
        t_next = min(t + dt, T)
        b = step_rhs(u, t_next, x, dt)
        u = solve_A(b)  # résolution via le solveur choisi
        t = t_next

        while next_store_idx < len(times_to_store) and t >= times_to_store[next_store_idx] - 1e-12:
            snapshots.append((times_to_store[next_store_idx], u.copy()))
            next_store_idx += 1

    return x, snapshots


# ==========================================================
# Lancement + visualisation
# ==========================================================
x, snapshots = solve_pde(L, v, nu, lam, T, N, dt, solver=SOLVER)

plt.figure()
for (ti, ui) in snapshots:
    plt.plot(x, ui, label=f"t={ti:.3f}")
plt.xlabel("x")
plt.ylabel("u(t,x)")
plt.title(f"Convection–Diffusion–Réaction 1D — solveur: {SOLVER}")
plt.legend()
plt.tight_layout()
plt.savefig("Convection_Diffusion_Reaction_1D.png", dpi=300, bbox_inches="tight")
plt.show()
