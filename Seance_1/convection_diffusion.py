"""
Convection–Diffusion(–Réaction) 2D (Dirichlet uniquement aux bords entrants).
- Génère **uniquement** un triptyque (collage) des 3 vues :
  solution, erreur sur u, erreur sur la norme du gradient.
- Sauvegarde **dans le même dossier que ce script** et affiche le triptyque.

Équation : u_t + V·∇u − ν Δu = −λ u + f(x,y),
           f(x,y) = Tc · exp(−k · ‖(x,y) − s_c‖²).
"""

from io import BytesIO
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def bords_entrants(v1, v2):
    """Côtés entrants (V·n < 0) pour V=(v1,v2)."""
    return (v1 > 0), (v1 < 0), (v2 > 0), (v2 < 0)  # gauche, droite, bas, haut

def masque_dirichlet(Nx, Ny, inflow):
    in_left, in_right, in_bot, in_top = inflow
    m = np.zeros((Ny, Nx), dtype=bool)
    if in_left:  m[:, 0]  = True
    if in_right: m[:, -1] = True
    if in_bot:   m[0, :]  = True
    if in_top:   m[-1, :] = True
    return m

def assemble_operateur(Nx, Ny, dx, dy, dt, nu, lam, mD):
    alpha = 1.0/dt + lam
    N = Nx*Ny
    rows, cols, vals = [], [], []

    def idg(i,j): return i + Nx*j

    for j in range(Ny):
        for i in range(Nx):
            p = idg(i,j)
            if mD[j,i]:
                rows.append(p); cols.append(p); vals.append(1.0)
                continue

            diag = alpha
            # x
            if i == 0:
                diag += nu*(1.0/dx**2)
                rows.append(p); cols.append(idg(i+1,j)); vals.append(-nu*(1.0/dx**2))
            else:
                rows.append(p); cols.append(idg(i-1,j)); vals.append(-nu*(1.0/dx**2))
                diag += nu*(1.0/dx**2)
            if i == Nx-1:
                diag += nu*(1.0/dx**2)
                rows.append(p); cols.append(idg(i-1,j)); vals.append(-nu*(1.0/dx**2))
            else:
                rows.append(p); cols.append(idg(i+1,j)); vals.append(-nu*(1.0/dx**2))
                diag += nu*(1.0/dx**2)
            # y
            if j == 0:
                diag += nu*(1.0/dy**2)
                rows.append(p); cols.append(idg(i,j+1)); vals.append(-nu*(1.0/dy**2))
            else:
                rows.append(p); cols.append(idg(i,j-1)); vals.append(-nu*(1.0/dy**2))
                diag += nu*(1.0/dy**2)
            if j == Ny-1:
                diag += nu*(1.0/dy**2)
                rows.append(p); cols.append(idg(i,j-1)); vals.append(-nu*(1.0/dy**2))
            else:
                rows.append(p); cols.append(idg(i,j+1)); vals.append(-nu*(1.0/dy**2))
                diag += nu*(1.0/dy**2)

            rows.append(p); cols.append(p); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

def advection_amont(u, v1, v2, dx, dy, uL, uR, uB, uT):
    Ny, Nx = u.shape
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    # x
    if v1 >= 0:
        dudx[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
        dudx[:, 0]  = (u[:, 0]  - uL[:, 0]) / dx
    else:
        dudx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
        dudx[:, -1]  = (uR[:, -1] - u[:, -1]) / dx
    # y
    if v2 >= 0:
        dudy[1:, :] = (u[1:, :] - u[:-1, :]) / dy
        dudy[0,  :] = (u[0,  :] - uB[0, :]) / dy
    else:
        dudy[:-1, :] = (u[1:, :] - u[:-1, :]) / dy
        dudy[-1,  :] = (uT[-1, :] - u[-1, :]) / dy
    return -(v1*dudx + v2*dudy)

def source_gauss(x, y, Tc, k, sc):
    X, Y = np.meshgrid(x, y, indexing='xy')
    return Tc * np.exp(-k * ((X - sc[0])**2 + (Y - sc[1])**2))

def norme_grad(u, dx, dy):
    Ny, Nx = u.shape
    dudx = np.zeros_like(u); dudy = np.zeros_like(u)
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*dy)
    dudx[:, 0]  = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dudy[0, :]  = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy
    return np.sqrt(dudx**2 + dudy**2)

def erreur_L2(champ, ref, dx, dy):
    diff = champ - ref
    return np.sqrt(np.sum(diff**2) * dx * dy)

def resout_imex(ax,bx,ay,by,Nx,Ny,T,v1,v2,nu,lam,u_in,Tc,k,sc,cfl):
    x = np.linspace(ax, bx, Nx)
    y = np.linspace(ay, by, Ny)
    dx = (bx-ax)/(Nx-1); dy = (by-ay)/(Ny-1)

    inflow = bords_entrants(v1, v2)
    mD = masque_dirichlet(Nx, Ny, inflow)

    limites = []
    if abs(v1)>0: limites.append(dx/abs(v1))
    if abs(v2)>0: limites.append(dy/abs(v2))
    dt = cfl*min(limites) if limites else 0.02*min(dx,dy)
    nsteps = int(np.ceil(T/dt)); dt = T/nsteps

    M = assemble_operateur(Nx, Ny, dx, dy, dt, nu, lam, mD)
    lu = spla.splu(M.tocsc())

    f = source_gauss(x, y, Tc, k, sc)
    u = np.zeros((Ny, Nx))

    uL = np.full((Ny, 1), u_in)
    uR = np.full((Ny, 1), u_in)
    uB = np.full((1, Nx), u_in)
    uT = np.full((1, Nx), u_in)

    for _ in range(nsteps):
        adv = advection_amont(u, v1, v2, dx, dy,
                              np.repeat(uL, Nx, axis=1)[:, :1],
                              np.repeat(uR, Nx, axis=1)[:, -1:],
                              np.repeat(uB, Ny, axis=0)[:1, :],
                              np.repeat(uT, Ny, axis=0)[-1:, :])
        u_star = u + dt*(adv + f)
        rhs = (u_star/dt).ravel()
        rhs[mD.ravel()] = u_in
        u = lu.solve(rhs).reshape(Ny, Nx)

    return x, y, u, {"dt": dt, "nsteps": nsteps}

def figure_as_image(plotter, figsize=(6,5), dpi=160):
    """Crée une figure Matplotlib via la fonction plotter(ax) et renvoie son image PIL en mémoire."""
    import PIL.Image as Image
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def collage_horizontal(images, outpath):
    """Colle des images PIL horizontalement et enregistre le résultat."""
    from PIL import Image
    h = max(im.size[1] for im in images)
    imgs = [im.resize((int(im.size[0]*h/im.size[1]), h)) for im in images]
    W = sum(im.size[0] for im in imgs)
    canvas = Image.new("RGB", (W, h), (255,255,255))
    xoff = 0
    for im in imgs:
        canvas.paste(im, (xoff, 0)); xoff += im.size[0]
    canvas.save(outpath)
    return canvas

def run():
    # ----- Paramètres -----
    ax, bx, ay, by = 0.0, 1.0, 0.0, 1.0
    Nx, Ny = 61, 61
    T = 1.0
    v1, v2 = 1.0, 0.3
    nu, lam = 0.01, 0.0
    u_in = 0.0
    Tc, k = 1.0, 80.0
    sc = (0.35, 0.55)
    cfl = 0.45

    # >>> Enregistrer dans le **même dossier que le script** <<<
    outdir = Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "triple_panel.png"

    # Solve coarse
    x, y, uC, infoC = resout_imex(ax,bx,ay,by,Nx,Ny,T,v1,v2,nu,lam,u_in,Tc,k,sc,cfl)
    dxC, dyC = (bx-ax)/(Nx-1), (by-ay)/(Ny-1)

    # Reference fine
    NxF, NyF = 2*(Nx-1)+1, 2*(Ny-1)+1
    xF, yF, uF, infoF = resout_imex(ax,bx,ay,by,NxF,NyF,T,v1,v2,nu,lam,u_in,Tc,k,sc,cfl)
    uF_on_C = uF[::2, ::2]

    # Errors
    e_u_L2 = erreur_L2(uC, uF_on_C, dxC, dyC)
    gC = norme_grad(uC, dxC, dyC)
    dxF, dyF = (bx-ax)/(NxF-1), (by-ay)/(NyF-1)
    gF = norme_grad(uF, dxF, dyF)
    gF_on_C = gF[::2, ::2]
    e_g_L2 = erreur_L2(gC, gF_on_C, dxC, dyC)

    e_u_pw = np.abs(uC - uF_on_C)
    e_g_pw = np.abs(gC - gF_on_C)
    extent = [ax, bx, ay, by]

    # Figures en mémoire
    def plot_solution(axp):
        im = axp.imshow(uC, extent=extent, origin='lower', aspect='auto')
        axp.set_title(f"Solution u(x,y, T={T:.2f})\nV=({v1},{v2}), ν={nu}, λ={lam}")
        axp.set_xlabel("x"); axp.set_ylabel("y")
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

    def plot_err_u(axp):
        im = axp.imshow(e_u_pw, extent=extent, origin='lower', aspect='auto')
        axp.set_title(f"Erreur ponctuelle |u - u_ref| (‖e‖₂={e_u_L2:.2e})")
        axp.set_xlabel("x"); axp.set_ylabel("y")
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

    def plot_err_grad(axp):
        im = axp.imshow(e_g_pw, extent=extent, origin='lower', aspect='auto')
        axp.set_title(f"Erreur ponctuelle |‖∇u‖ - ‖∇u_ref‖| (‖e‖₂={e_g_L2:.2e})")
        axp.set_xlabel("x"); axp.set_ylabel("y")
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

    img1 = figure_as_image(plot_solution)
    img2 = figure_as_image(plot_err_u)
    img3 = figure_as_image(plot_err_grad)

    # Collage (unique fichier écrit)
    collage = collage_horizontal([img1, img2, img3], outfile)

    # Affichage
    plt.figure(figsize=(14,5))
    plt.imshow(collage)
    plt.axis("off")
    plt.title("Triptyque : Solution — Erreur u — Erreur ‖∇u‖")
    plt.show()

    print("Triptyque enregistré dans :", outfile)

if __name__ == "__main__":
    run()
