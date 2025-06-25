# python turbulence_angles_full.py --N 512 --outfile baseline.h5
# python turbulence_angles_full.py --N 256 --outfile baseline.h5

# python turbulence_angles_full.py --N 512 --solenoidal --outfile solenoidal.h5
# python turbulence_angles_full.py --N 256 --solenoidal --outfile solenoidal.h5

import argparse, os, math
from pathlib import Path
import h5py
import numpy as np
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.stats import linregress

def _helmholtz_project(u_k, kx, ky, kz):
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.     # avoid /0
    k_dot_u = kx*u_k[...,0] + ky*u_k[...,1] + kz*u_k[...,2]
    u_k[...,0] -= kx * k_dot_u / k2
    u_k[...,1] -= ky * k_dot_u / k2
    u_k[...,2] -= kz * k_dot_u / k2
    return u_k


def generate_velocity_cube(N=256, outer_scale=None,
                           solenoidal=False, seed=None):
    rng = np.random.default_rng(seed)
    outer_scale = outer_scale or N/4

    kx = fftfreq(N)[:,None,None]
    ky = fftfreq(N)[None,:,None]
    kz = rfftfreq(N)[None,None,:]
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    kmag[0,0,0] = 1.

    amp = kmag**(-11/6)
    amp *= np.exp(-(kmag*outer_scale/N)**2)     # large‑scale cut‑off

    phase = rng.standard_normal((*amp.shape, 3)) \
          + 1j*rng.standard_normal((*amp.shape, 3))
    u_k = amp[...,None] * phase

    if solenoidal:
        u_k = _helmholtz_project(u_k, kx, ky, kz)

    u_real = irfftn(u_k, s=(N,N,N), axes=(0,1,2)).astype(np.float32)
    return u_real


def los_integrate(u, axis=2):
    """∫ u ds  along LOS axis."""
    return np.sum(u, axis=axis)

def angle_azimuth(u_int):
    ux, uy = u_int[...,0], u_int[...,1]
    return np.arctan2(uy, ux)

@njit(fastmath=True, parallel=True)
def _stokes_accumulate(u):
    """Sum Q,U per cell along z (Numba parallel)."""
    Nx, Ny, Nz, _ = u.shape
    Q = np.zeros((Nx, Ny), np.float32)
    U = np.zeros((Nx, Ny), np.float32)
    for i in prange(Nx):
        for j in prange(Ny):
            q = 0.; u_loc = 0.
            for k in range(Nz):
                ux, uy = u[i,j,k,0], u[i,j,k,1]
                phi = math.atan2(uy, ux)
                q += math.cos(2*phi)
                u_loc += math.sin(2*phi)
            Q[i,j] = q; U[i,j] = u_loc
    return Q, U

def angle_stokes(u):
    Q, U = _stokes_accumulate(u)
    return 0.5*np.arctan2(U, Q)

def polar_angle(u):
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    return np.arctan2(np.sqrt(ux**2 + uy**2), uz)

def angle_between(v1, v2):
    dot = (v1*v2).sum(axis=-1)
    n1  = np.linalg.norm(v1, axis=-1)
    n2  = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot/(n1*n2 + 1e-30), -1., 1.)
    return np.arccos(cos)

def angular_difference(phi1, phi2):
    return (phi2 - phi1 + np.pi) % (2*np.pi) - np.pi

def sample_pairs_2d(field, R, max_pairs=3_00_000, rng=None):
    Nx, Ny = field.shape
    rng = rng or np.random.default_rng()
    dirs = [(R,0),(-R,0),(0,R),(0,-R),(R,R),(-R,-R),(R,-R),(-R,R)]
    out = []
    for dx,dy in dirs:
        n = int(np.sqrt(max_pairs//len(dirs)))*1000
        xs = rng.integers(0,Nx,size=n)
        ys = rng.integers(0,Ny,size=n)
        p1 = field[xs, ys]
        p2 = field[(xs+dx)%Nx, (ys+dy)%Ny]
        out.append(angular_difference(p1, p2))
    return np.concatenate(out)

def sample_pairs_3d(u, R, max_pairs=2_00_000, rng=None):
    Nx, Ny, Nz, _ = u.shape
    rng = rng or np.random.default_rng()
    dirs = [(R,0,0),(-R,0,0),(0,R,0),(0,-R,0),(0,0,R),(0,0,-R)]
    out=[]
    for dx,dy,dz in dirs:
        n = int(round((max_pairs/len(dirs))**(1/3)))*1000
        xs=rng.integers(0,Nx,n); ys=rng.integers(0,Ny,n); zs=rng.integers(0,Nz,n)
        v1=u[xs,ys,zs]; v2=u[(xs+dx)%Nx,(ys+dy)%Ny,(zs+dz)%Nz]
        out.append(angle_between(v1,v2))
    return np.concatenate(out)

def pdf_1d(samples, bins):
    counts, edges = np.histogram(samples, bins=bins, density=False)
    centers = 0.5*(edges[:-1]+edges[1:])
    pdf = counts / counts.sum() / (edges[1]-edges[0])
    return centers, pdf

def structure_function_2d(field, Rs, max_pairs=3_00_000):
    D=[]
    for R in Rs:
        dphi = sample_pairs_2d(field,R,max_pairs=max_pairs)
        D.append(0.5*(1 - np.cos(2*dphi).mean()))
    return np.asarray(Rs), np.asarray(D)

def structure_function_3d(u, Rs, max_pairs=2_00_000):
    D=[]
    for R in Rs:
        dth = sample_pairs_3d(u,R,max_pairs=max_pairs)
        D.append(0.5*(1 - np.cos(dth).mean()))
    return np.asarray(Rs), np.asarray(D)


def ensure_dir(path="figures"):
    Path(path).mkdir(exist_ok=True)
    return Path(path)

def plot_single_point(field, kind, outdir):
    if kind=="polar":
        bins=np.linspace(0,np.pi,91)
        weight=np.sin  # Jacobian
    else:  # azimuth
        bins=np.linspace(-np.pi,np.pi,181)
        weight=lambda x:1.
    x,pdf=pdf_1d(field.ravel(),bins)
    pdf/=weight(x)         
    plt.figure()
    plt.semilogy(x,pdf,'.')
    plt.xlabel(r'$\theta$' if kind=="polar" else r'$\varphi$')
    plt.ylabel(r'$P_{1}$')
    plt.title(f'Single‑point PDF ({kind})')
    plt.tight_layout()
    plt.savefig(outdir/f'pdf_single_{kind}.pdf',dpi=200)
    plt.close()

def plot_two_point(field_or_u, Rs, mode, outdir):
    plt.figure()
    slopes={'vector':2/3,'stokes':2/3,'polar3d':2/3}
    if mode=='polar3d':
        sampler=lambda R:sample_pairs_3d(field_or_u,R)
        xlabel=r'$\Delta\theta$'
        bins=np.linspace(0,np.pi,91)
        weight=lambda x:np.sin(x)
    else:
        sampler=lambda R:sample_pairs_2d(field_or_u,R)
        xlabel=r'$\Delta\varphi$'
        bins=np.linspace(-np.pi,np.pi,181)
        weight=lambda x:1.
    for R in Rs:
        samp=sampler(R)
        x,pdf=pdf_1d(samp,bins)
        pdf/=weight(x)
        plt.semilogy(x,pdf,label=f'R={R}')
    plt.xlabel(xlabel); plt.ylabel(r'$P_{\Delta}$')
    txt={'vector':'Vector azimuth','stokes':'Stokes azimuth',
         'polar3d':'3‑D polar'}
    plt.title(txt[mode])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/f'pdf_pairs_{mode}.pdf',dpi=200)
    plt.close()

    if mode=='polar3d':
        R_arr,D=structure_function_3d(field_or_u,Rs)
    else:
        R_arr,D=structure_function_2d(field_or_u,Rs)
    slope=slopes[mode]
    plt.figure()
    
    half = len(R_arr) # len(R_arr) // 2
    lsm_slope, lsm_intercept, _, _, lsm_std_err = linregress(np.log(R_arr[:half]), np.log(D[:half]))
    
    plt.loglog(R_arr, D, 'o', label=fr'simulation (slope = ${lsm_slope:.2f} \pm {lsm_std_err:.2f}$)')
    C = D[1] / R_arr[1]**slope

    plt.loglog(R_arr,C*R_arr**slope,'--', # ref = 0.1*R_arr**slope
               label=fr'$R^{{{slope:.2f}}}$')

    plt.xlabel('R'); plt.ylabel('D(R)')
    plt.title(f'Structure function – {txt[mode]}')
    plt.ylim(1e-3, 1e0)
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir/f'structure_{mode}.pdf',dpi=200); plt.close()


def main():
    ap=argparse.ArgumentParser(
        description="Synthetic Kolmogorov turbulence → full angle statistics")
    ap.add_argument("--N",type=int,default=256)
    ap.add_argument("--los-axis",type=int,choices=[0,1,2],default=2)
    ap.add_argument("--solenoidal",action='store_true')
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--outer-scale",type=float,default=None)
    ap.add_argument("--outfile",default="simulation.h5")
    args=ap.parse_args()

    print(f"Generating cube  N={args.N}  solenoidal={args.solenoidal}")
    u=generate_velocity_cube(N=args.N,
                             outer_scale=args.outer_scale,
                             solenoidal=args.solenoidal,
                             seed=args.seed)
    print("LOS integrate …")
    u_int=los_integrate(u,axis=args.los_axis)
    theta_vec=angle_azimuth(u_int)
    theta_stokes=angle_stokes(u)

    outdir=ensure_dir("figures")
    print("Single‑point PDFs …")
    plot_single_point(polar_angle(u),"polar",outdir)
    plot_single_point(theta_vec,"azimuth_vec",outdir)  

    (outdir/"pdf_single_azimuth_vec.pdf").rename(outdir/"pdf_single_vector.pdf")
    plot_single_point(theta_stokes,"azimuth_stokes",outdir)
    (outdir/"pdf_single_azimuth_stokes.pdf").rename(outdir/"pdf_single_stokes.pdf")

    Rs=np.unique(np.logspace(0,np.log10(args.N//3),20).astype(int))
    print("Two‑point statistics …")
    plot_two_point(theta_vec,Rs,"vector",outdir)
    plot_two_point(theta_stokes,Rs,"stokes",outdir)
    plot_two_point(u,Rs,"polar3d",outdir)

    print(f"Saving data → {args.outfile}")
    with h5py.File(args.outfile,'w') as h5:
        h5["u"]=u; h5["u_int"]=u_int
        h5["theta_vector"]=theta_vec; h5["theta_stokes"]=theta_stokes
        h5["polar_angle"]=polar_angle(u)
        h5.attrs["N"]=args.N
        h5.attrs["solenoidal"]=args.solenoidal
        h5.attrs["seed"]=args.seed if args.seed is not None else -1
    print("Done.  Figures in ./figures/")

if __name__=="__main__":
    main()
