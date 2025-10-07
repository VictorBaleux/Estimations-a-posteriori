
import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# u(t) and its derivative
# -----------------------------
omega = 4.0 * math.pi

def U(t):
    return math.sin(omega * t)

def Up(t):
    return omega * math.cos(omega * t)

# -----------------------------
# Controls for visualization
# -----------------------------
Time = 1.0  # total simulation time [s]
times_to_plot = [0.0, 0.25, 0.50, 0.75, 1.00]  # instants to visualize

iplot = 0  # set 1 to print inline time plots during run (slower)

# -----------------------------
# PHYSICAL PARAMETERS
# -----------------------------
K = 0.01     # Diffusion coefficient
xmin = 0.0
xmax = 1.0
V = 1.0
lamda = 1.0

# -----------------------------
# Mesh adaptation parameters
# -----------------------------
niter_refinement = 30     # hard cap safeguard
hmin = 0.01
hmax = 0.15
err_metric = 0.03  # target for second derivative in metric

# MIXED stopping criterion (do NOT stop until BOTH are satisfied)
MAX_POINTS = 250          # criterion on number of mesh points
L2_TOL = 1e-3             # criterion on L2 error wrt U(t_end)*Tex

# -----------------------------
# NUMERICAL PARAMETERS
# -----------------------------
NX_init = 3     # initial number of grid points
NT = 100000     # max time steps per adaptation cycle
ifre = 1000000
eps_rel_res = 1e-3  # relative residual threshold (diagnostic)

# -----------------------------
# Storage for error vs adaptation
# -----------------------------
errorL2 = np.zeros((niter_refinement))
errorH1 = np.zeros((niter_refinement))
itertab = np.zeros((niter_refinement))

# For background Cauchy check (optional)
NX_background = 200
background_mesh = np.linspace(xmin, xmax, NX_background)
Tbacknew = []

# For snapshots across adaptation (we plot them from the last adaptation)
snapshots = {}  # time -> (x, T_at_time)

def build_Tex(x, xmin, xmax):
    Tex = np.zeros_like(x)
    for j in range(1, len(x) - 1):
        Tex[j] = 2.0 * np.exp(-100.0 * (x[j] - (xmax + xmin) * 0.25) ** 2) \
               + 1.0 * np.exp(-200.0 * (x[j] - (xmax + xmin) * 0.65) ** 2)
    Tex[0] = Tex[1]
    Tex[-1] = Tex[-2]
    return Tex

def second_derivative_nonuniform(y, x, j):
    Txip1 = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
    Txim1 = (y[j] - y[j - 1]) / (x[j] - x[j - 1])
    denom = (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1]))
    return (Txip1 - Txim1) / denom

def first_derivative_central(y, x, j):
    return (y[j + 1] - y[j - 1]) / (x[j + 1] - x[j - 1])

def interpolate_piecewise_linear(x_from, y_from, x_query):
    yq = np.empty_like(x_query)
    i = 0
    for k, xq in enumerate(x_query):
        while i < len(x_from) - 2 and xq > x_from[i + 1]:
            i += 1
        xL, xR = x_from[i], x_from[i + 1]
        yL, yR = y_from[i], y_from[i + 1]
        if xR == xL:
            yq[k] = yL
        else:
            yq[k] = (yR * (xq - xL) + yL * (xR - xq)) / (xR - xL)
    return yq

# -----------------------------
# Initialize mesh & metric to carry across adaptation cycles
# -----------------------------
x_prev = np.linspace(xmin, xmax, NX_init)
hloc_prev = np.ones_like(x_prev) * (hmax * 0.5)  # initial quasi-uniform metric
T_prev = np.zeros_like(x_prev)  # for interpolation if desired

itera = 0

# Run adaptation cycles until MIXED criterion is satisfied
while True:
    if itera >= niter_refinement:
        print("Reached niter_refinement cap =", niter_refinement)
        break

    itertab[itera] = 1.0 / max(len(x_prev), 1)

    # --- mesh adaptation using hloc_prev on x_prev ---
    xnew = [xmin]
    Tnew = [0.0]  # start from zero state on new mesh; could also interpolate T_prev if wanted
    nnew = 1
    while xnew[nnew - 1] < xmax - hmin:
        # find containing interval in previous mesh
        for i in range(len(x_prev) - 1):
            if xnew[nnew - 1] >= x_prev[i] and xnew[nnew - 1] <= x_prev[i + 1] and xnew[nnew - 1] < xmax - hmin:
                # linear interp of previous metric hloc_prev between x_prev[i], x_prev[i+1]
                hll = (hloc_prev[i] * (x_prev[i + 1] - xnew[nnew - 1]) + hloc_prev[i + 1] * (xnew[nnew - 1] - x_prev[i])) / (x_prev[i + 1] - x_prev[i])
                hll = min(max(hmin, hll), hmax)
                nnew += 1
                xnew.append(min(xmax, xnew[nnew - 2] + hll))

                # here we keep Tnew ~ 0; if you want to interpolate from previous T, uncomment:
                # un = (T_prev[i] * (x_prev[i + 1] - xnew[nnew - 1]) + T_prev[i + 1] * (xnew[nnew - 1] - x_prev[i])) / (x_prev[i + 1] - x_prev[i])
                # Tnew.append(un)
                Tnew.append(0.0)

    x = np.array(xnew[:nnew])
    T = np.array(Tnew[:nnew])
    NX = len(x)

    # --- arrays for this cycle ---
    rest = []
    F = np.zeros((NX))
    RHS = np.zeros((NX))
    hloc = np.ones((NX)) * (hmax * 0.5)  # will be updated from metric
    metric = np.ones((NX))

    # spatial profile v(x) and its operators (time-independent)
    Tex = build_Tex(x, xmin, xmax)

    # Precompute G1(x) = V*v'(x) - K*v''(x) + lamda*v(x)
    G1 = np.zeros((NX))
    Fbound = np.zeros((NX))
    dt = 1.0e30
    for j in range(1, NX - 1):
        Tx = first_derivative_central(Tex, x, j)
        Txx = second_derivative_nonuniform(Tex, x, j)
        G1[j] = V * Tx - K * Txx + lamda * Tex[j]

        # bound for source to choose a stable dt (very conservative)
        Fbound[j] = abs(G1[j]) + omega * abs(Tex[j])

        dxloc = (x[j + 1] - x[j - 1])
        dt = min(dt, 0.5 * dxloc ** 2 / (V * abs(dxloc) + 4.0 * K + Fbound[j] * dxloc ** 2))

    G1[0] = G1[1]
    G1[-1] = G1[-2]

    print(f'[Adapt {itera+1}] NX={NX}, chosen dt={dt:.3e}')

    # --- time stepping on current mesh ---
    n = 0
    res = 1.0
    res0 = 1.0
    t = 0.0

    # prepare snapshot bookkeeping
    next_plot_id = 0
    if abs(times_to_plot[0] - 0.0) < 1e-12:
        snapshots[times_to_plot[0]] = (x.copy(), T.copy())
        next_plot_id = 1

    while n < NT and res / res0 > eps_rel_res and t < Time - 1e-15:
        n += 1
        t_next = min(t + dt, Time)
        dt_eff = t_next - t
        t = t_next

        # residual accumulation
        res = 0.0
        for j in range(1, NX - 1):
            # numerical viscosity for upwinding
            visnum = 0.5 * (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1])) * abs(V)
            xnu = K + visnum

            # spatial derivatives of T
            TxT = first_derivative_central(T, x, j)
            TxxT = second_derivative_nonuniform(T, x, j)

            # time-dependent source: f(t,x) = U'(t)*Tex + U(t)*G1
            Ftj = Up(t) * Tex[j] + U(t) * G1[j]

            RHS[j] = dt_eff * (-V * TxT + xnu * TxxT - lamda * T[j] + Ftj)

            # metric for adaptation
            metric[j] = min(1.0 / hmin ** 2, max(1.0 / hmax ** 2, abs(TxxT) / err_metric))

            res += abs(RHS[j])

        metric[0] = metric[1]
        metric[-1] = metric[-2]

        # smooth metric (edge-avg)
        for j in range(0, NX - 1):
            metric[j] = 0.5 * (metric[j] + metric[j + 1])
        metric[-1] = metric[-2]

        hloc[:] = np.sqrt(1.0 / metric[:])

        # explicit update
        for j in range(1, NX - 1):
            T[j] += RHS[j]
            RHS[j] = 0.0

        # right boundary (Neumann-like)
        T[-1] = T[-2]

        if n == 1:
            res0 = max(res, 1e-30)
        rest.append(res)

        # take snapshots at requested times
        while next_plot_id < len(times_to_plot) and t >= times_to_plot[next_plot_id] - 1e-12:
            snapshots[times_to_plot[next_plot_id]] = (x.copy(), T.copy())
            next_plot_id += 1

        if (n % ifre == 0) or (res / res0) < eps_rel_res:
            print(f'  time step {n}, t={t:.4f}, residual={res:.3e}')

    # --- background Cauchy check (optional) ---
    Tbackold = Tbacknew.copy()
    Tbacknew = interpolate_piecewise_linear(x, T, background_mesh)
    if len(Tbackold) == len(Tbacknew):
        cauchy = float(np.sum(np.abs(Tbacknew - Tbackold)))
        print("  Cauchy background diff =", cauchy)

    # --- error vs analytical separable profile at final time ---
    # Compare to U(t_end)*Tex
    errH1h = 0.0
    errL2h = 0.0
    for j in range(1, NX - 1):
        Texx = first_derivative_central(Tex, x, j)
        TxT = first_derivative_central(T, x, j)
        w = (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1]))
        errL2h += w * (T[j] - U(t) * Tex[j]) ** 2
        errH1h += w * (TxT - U(t) * Texx) ** 2

    errorL2[itera] = errL2h
    errorH1[itera] = errL2h + errH1h

    print(f'  [Adapt {itera+1}] L2={errL2h:.3e}, H1_add={errH1h:.3e}, NX={NX}, t_end={t:.4f}')
    print('----------------------------------')

    # --- MIXED STOPPING CHECK ---
    both_satisfied = (NX <= MAX_POINTS) and (errorL2[itera] <= L2_TOL)

    # carry current mesh & metric to next cycle
    x_prev = x.copy()
    hloc_prev = hloc.copy()
    T_prev = T.copy()

    itera += 1
    if both_satisfied:
        break

# -----------------------------
# Post-processing: plots
# -----------------------------
for tt in times_to_plot:
    if tt in snapshots:
        xplt, Tplt = snapshots[tt]
        plt.figure()
        plt.plot(xplt, Tplt, marker='o')
        plt.title(f'Snapshot at t = {tt:.2f} s')
        plt.xlabel('x')
        plt.ylabel('T')
        plt.grid(True)
        plt.tight_layout()

if itera >= 2:
    ii = np.arange(itera)
    plt.figure()
    # number of points is len(x_prev) at last cycle; for history we used proxy 1/itertab
    plt.plot(1.0 / itertab[0:itera], errorL2[0:itera], marker='o')
    plt.xlabel('Number of points (proxy: 1 / itertab)')
    plt.ylabel('L2 error at t = Time')
    plt.title('Adaptation convergence (L2 vs #points)')
    plt.grid(True)
    plt.tight_layout()

plt.show()
