
"""
Convection–Diffusion(–Réaction) 2D
----------------------------------
- Diffusion : conditions de Neumann homogènes (∂u/∂n = 0) SUR TOUS LES BORDS,
  traitées dans l'opérateur implicite via des stencils à point fantôme (facteur 2 sur le voisin intérieur).
- Advection : schéma d'amont (upwind) explicite. Les valeurs aux bords amont sont
  passées **directement** via uL (gauche, taille Ny), uR (droite, Ny), uB (bas, Nx), uT (haut, Nx).
  Pas de `repeat` ni de masques Dirichlet : l'amont impose les valeurs d'entrée.
- Réaction : terme -λ u implicite (lumped dans la diagonale).
- Sortie : un triptyque (solution, erreur u, erreur ‖∇u‖) et une courbe de convergence spatiale
  ‖e(u)‖₂ et ‖e(∇u)‖₂ en fonction de h sur 2–3 raffinements (attendu ≈ ordre 1 avec amont + temps ordre 1).

Équation : u_t + V·∇u − ν Δu = −λ u + f(x,y),
           f(x,y) = Tc · exp(−k · ‖(x,y) − s_c‖²).
"""

from io import BytesIO
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --------------------- Utilitaires PDE ---------------------

def assemble_operateur(Nx, Ny, dx, dy, dt, nu, lam):
    """
    Assemble M ≈ (I/dt + lam) - nu * Δ  avec Neumann homogène sur le bord.
    Discrétisation en différences finies sur grille cartésienne (indexing 'xy').
    Aux bords : stencil à point fantôme ⇒ coefficient *2 sur le voisin intérieur.
    """
    alpha = 1.0/dt + lam
    N = Nx*Ny
    rows, cols, vals = [], [], []

    def idg(i,j): return i + Nx*j

    cx = nu/(dx*dx) if Nx > 1 else 0.0
    cy = nu/(dy*dy) if Ny > 1 else 0.0

    for j in range(Ny):
        for i in range(Nx):
            p = idg(i,j)
            diag = alpha

            # --- x-direction with homogeneous Neumann ---
            if Nx > 1:
                if i == 0:
                    # u_{-1} = u_{1}  ⇒  Δx u_0 ≈ 2(u_1 - u_0)/dx²
                    diag += 2*cx
                    rows.append(p); cols.append(idg(i+1, j)); vals.append(-2*cx)
                elif i == Nx-1:
                    diag += 2*cx
                    rows.append(p); cols.append(idg(i-1, j)); vals.append(-2*cx)
                else:
                    diag += 2*cx
                    rows.append(p); cols.append(idg(i-1, j)); vals.append(-cx)
                    rows.append(p); cols.append(idg(i+1, j)); vals.append(-cx)

            # --- y-direction with homogeneous Neumann ---
            if Ny > 1:
                if j == 0:
                    diag += 2*cy
                    rows.append(p); cols.append(idg(i, j+1)); vals.append(-2*cy)
                elif j == Ny-1:
                    diag += 2*cy
                    rows.append(p); cols.append(idg(i, j-1)); vals.append(-2*cy)
                else:
                    diag += 2*cy
                    rows.append(p); cols.append(idg(i, j-1)); vals.append(-cy)
                    rows.append(p); cols.append(idg(i, j+1)); vals.append(-cy)

            rows.append(p); cols.append(p); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

def advection_amont(u, v1, v2, dx, dy, uL, uR, uB, uT):
    """
    Flux d'advection explicite (amont).
    Entrée :
      - u : (Ny, Nx)
      - v1, v2 : vitesses constantes V=(v1,v2)
      - uL (Ny,), uR (Ny,), uB (Nx,), uT (Nx,)
        valeurs aux bords gauche/droite/bas/haut UTILISÉES UNIQUEMENT
        quand la vitesse entre dans le domaine par ce bord.
    Sortie : - (v1 ∂x u + v2 ∂y u) évalué par différences amont.
    """
    Ny, Nx = u.shape
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)

    # x-direction
    if v1 >= 0:
        # amont = arrière
        if Nx > 1:
            dudx[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
        # bord gauche utilise uL
        dudx[:, 0] = (u[:, 0] - uL) / dx
    else:
        # amont = avant
        if Nx > 1:
            dudx[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
        # bord droit utilise uR
        dudx[:, -1] = (uR - u[:, -1]) / dx

    # y-direction
    if v2 >= 0:
        if Ny > 1:
            dudy[1:, :] = (u[1:, :] - u[:-1, :]) / dy
        dudy[0, :] = (u[0, :] - uB) / dy
    else:
        if Ny > 1:
            dudy[:-1, :] = (u[1:, :] - u[:-1, :]) / dy
        dudy[-1, :] = (uT - u[-1, :]) / dy

    return -(v1 * dudx + v2 * dudy)

def source_gauss(x, y, Tc, k, sc):
    X, Y = np.meshgrid(x, y, indexing='xy')
    return Tc * np.exp(-k * ((X - sc[0])**2 + (Y - sc[1])**2))

def norme_grad(u, dx, dy):
    Ny, Nx = u.shape
    dudx = np.zeros_like(u); dudy = np.zeros_like(u)
    if Nx > 1:
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)
        dudx[:, 0]  = (u[:, 1] - u[:, 0]) / dx
        dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    if Ny > 1:
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*dy)
        dudy[0, :]  = (u[1, :] - u[0, :]) / dy
        dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy
    return np.sqrt(dudx**2 + dudy**2)

def erreur_L2(champ, ref, dx, dy):
    diff = champ - ref
    return np.sqrt(np.sum(diff**2) * dx * dy)

# --------------------- Solveur IMEX ---------------------

def resout_imex(ax,bx,ay,by,Nx,Ny,T,v1,v2,nu,lam,uL,uR,uB,uT,Tc,k,sc,cfl):
    x = np.linspace(ax, bx, Nx)
    y = np.linspace(ay, by, Ny)
    dx = (bx-ax)/(Nx-1) if Nx>1 else 1.0
    dy = (by-ay)/(Ny-1) if Ny>1 else 1.0

    # CFL simple basé sur l'advection
    limites = []
    if abs(v1)>0 and Nx>1: limites.append(dx/abs(v1))
    if abs(v2)>0 and Ny>1: limites.append(dy/abs(v2))
    dt = cfl*min(limites) if limites else 0.02*min(dx,dy)  # fallback
    nsteps = int(np.ceil(T/dt)); dt = T/nsteps

    M = assemble_operateur(Nx, Ny, dx, dy, dt, nu, lam)
    lu = spla.splu(M.tocsc())

    f = source_gauss(x, y, Tc, k, sc)
    u = np.zeros((Ny, Nx))

    for _ in range(nsteps):
        adv = advection_amont(u, v1, v2, dx, dy, uL, uR, uB, uT)
        u_star = u + dt*(adv + f)
        rhs = (u_star/dt).ravel()
        u = lu.solve(rhs).reshape(Ny, Nx)

    info = {"dt": dt, "nsteps": nsteps}
    return x, y, u, info

# --------------------- Figures ---------------------

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

def convergence_spatiale(ax,bx,ay,by,N0,levels,T,v1,v2,nu,lam,u_in,Tc,k,sc,cfl,outpath):
    """
    Calcule les erreurs L2 de u et ‖∇u‖ en fonction de h sur une hiérarchie emboîtée :
    N0 -> N1=2(N0-1)+1 -> N2=2(N1-1)+1 -> ...
    Le niveau le plus fin sert de référence.
    """
    # Construire la liste des N
    Ns = [N0]
    for _ in range(1, levels):
        Ns.append(2*(Ns[-1]-1)+1)

    # Résolutions pour tous les niveaux
    sols = []
    dxy = []
    for N in Ns:
        Nx = Ny = N
        # valeurs d'entrée (amont) constantes = u_in ; on peut remplacer par profils si besoin
        uL = np.full(Ny, u_in)
        uR = np.full(Ny, u_in)
        uB = np.full(Nx, u_in)
        uT = np.full(Nx, u_in)
        x, y, u, info = resout_imex(ax,bx,ay,by,Nx,Ny,T,v1,v2,nu,lam,uL,uR,uB,uT,Tc,k,sc,cfl)
        dx = (bx-ax)/(Nx-1); dy = (by-ay)/(Ny-1)
        sols.append(u)
        dxy.append(max(dx,dy))

    # Référence = plus fin
    uF = sols[-1]
    NF = Ns[-1]
    dxF = (bx-ax)/(NF-1); dyF = (by-ay)/(NF-1)
    from math import isclose

    e_u = []
    e_g = []
    hs  = []
    for N, uC, hC in zip(Ns[:-1], sols[:-1], dxy[:-1]):
        r = (NF-1)//(N-1)  # ratio entier (emboîtement garanti)
        assert (NF-1) % (N-1) == 0
        uF_on_C = uF[::r, ::r]

        # erreurs
        dxC = (bx-ax)/(N-1); dyC = (by-ay)/(N-1)
        e_u.append(erreur_L2(uC, uF_on_C, dxC, dyC))

        gC = norme_grad(uC, dxC, dyC)
        gF = norme_grad(uF, dxF, dyF)
        gF_on_C = gF[::r, ::r]
        e_g.append(erreur_L2(gC, gF_on_C, dxC, dyC))

        hs.append(hC)

    # fit pente
    logh = np.log(hs)
    logeu = np.log(e_u)
    logeg = np.log(e_g)
    pu = np.polyfit(logh, logeu, 1)[0]
    pg = np.polyfit(logh, logeg, 1)[0]

    # figure
    plt.figure(figsize=(6.5,5))
    plt.loglog(hs, e_u, marker='o', label=r"$\|e(u)\|_{L^2}$")
    plt.loglog(hs, e_g, marker='s', label=r"$\|e(\|\nabla u\|)\|_{L^2}$")
    # ligne de pente 1 pour repère
    c0 = e_u[0]/hs[0]  # normalise pour passer par (h0, e_u0)
    plt.loglog(hs, c0*np.array(hs), linestyle='--', label="pente 1 (réf)")
    plt.gca().invert_xaxis()
    plt.xlabel("pas h")
    plt.ylabel("erreur L2")
    plt.title(f"Convergence spatiale (p_u≈{pu:.2f}, p_grad≈{pg:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.show()
    return {"N": Ns, "h": hs, "e_u": e_u, "e_grad": e_g, "p_u": pu, "p_grad": pg}

def run():
    # ----- Paramètres par défaut -----
    ax, bx, ay, by = 0.0, 1.0, 0.0, 1.0
    Nx, Ny = 61, 61
    T = 1.0
    v1, v2 = 1.0, 0.3
    nu, lam = 0.01, 0.0
    u_in = 0.0
    Tc, k = 1.0, 80.0
    sc = (0.35, 0.55)
    cfl = 0.45

    # Dossier de sortie (même que le script)
    outdir = Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    outfile_trip = outdir / "triple_panel.png"
    outfile_conv = outdir / "convergence.png"

    # ---------- Résolution (grille "coarse") et référence fine ----------
    # Bords amont : valeurs constantes = u_in (peuvent être remplacées par profils)
    uL = np.full(Ny, u_in)
    uR = np.full(Ny, u_in)
    uB = np.full(Nx, u_in)
    uT = np.full(Nx, u_in)

    x, y, uC, infoC = resout_imex(ax,bx,ay,by,Nx,Ny,T,v1,v2,nu,lam,uL,uR,uB,uT,Tc,k,sc,cfl)
    dxC, dyC = (bx-ax)/(Nx-1), (by-ay)/(Ny-1)

    NxF, NyF = 2*(Nx-1)+1, 2*(Ny-1)+1
    uL_F = np.full(NyF, u_in)
    uR_F = np.full(NyF, u_in)
    uB_F = np.full(NxF, u_in)
    uT_F = np.full(NxF, u_in)
    xF, yF, uF, infoF = resout_imex(ax,bx,ay,by,NxF,NyF,T,v1,v2,nu,lam,uL_F,uR_F,uB_F,uT_F,Tc,k,sc,cfl)
    uF_on_C = uF[::2, ::2]

    # ---------- Erreurs ----------
    e_u_L2 = erreur_L2(uC, uF_on_C, dxC, dyC)
    gC = norme_grad(uC, dxC, dyC)
    dxF, dyF = (bx-ax)/(NxF-1), (by-ay)/(NyF-1)
    gF = norme_grad(uF, dxF, dyF)
    gF_on_C = gF[::2, ::2]
    e_g_L2 = erreur_L2(gC, gF_on_C, dxC, dyC)

    e_u_pw = np.abs(uC - uF_on_C)
    e_g_pw = np.abs(gC - gF_on_C)
    extent = [ax, bx, ay, by]

    # ---------- Triptyque ----------
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

    collage = collage_horizontal([img1, img2, img3], outfile_trip)

    # ---------- Convergence spatiale (3 niveaux par défaut) ----------
    conv = convergence_spatiale(ax,bx,ay,by,N0=41,levels=4,T=T,v1=v1,v2=v2,
                                nu=nu,lam=lam,u_in=u_in,Tc=Tc,k=k,sc=sc,cfl=cfl,
                                outpath=outfile_conv)

    # Affichage rapide (triptyque)
    plt.figure(figsize=(14,5))
    plt.imshow(collage)
    plt.axis("off")
    plt.title("Triptyque : Solution — Erreur u — Erreur ‖∇u‖")
    plt.show()

    print("Triptyque enregistré dans :", outfile_trip)
    print("Courbe de convergence enregistrée dans :", outfile_conv)
    print("Pentes observées : p_u ≈ {:.2f}, p_grad ≈ {:.2f}".format(conv["p_u"], conv["p_grad"]))


if __name__ == "__main__":
    run()
