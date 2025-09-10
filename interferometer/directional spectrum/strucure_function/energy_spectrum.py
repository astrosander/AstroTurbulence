#!/usr/bin/env python3
import os, h5py, numpy as np, matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift, fftfreq

H5="mhd_fields.h5"
# H5="two_slope_2D_s4_r00.h5"
DSET="k_mag_field"; NBINS=128; OUT="fig/energy_spectrum"; DPI=160

def _dx(a,f=1.0):
    try:
        u=np.unique(np.asarray(a).ravel()); d=np.diff(np.sort(u)); d=d[d>0]
        if d.size: return float(np.median(d))
    except: pass
    return float(f)

def load_field(p,d):
    with h5py.File(p,"r") as f:
        x=f[d][:].astype(float)
        dx=dy=_dx(f["x_coor"][:,0,0]) if "x_coor" in f else 1.0
        dz=_dx(f["z_coor"][0,0,:]) if "z_coor" in f else 1.0
    # print(dy)
    dx=1.0
    dy=1.0
    dz=1.0
    return x,(dx,dy,dz)

def spectrum(x,spc,n=128):
    x=x-x.mean(); F=fftn(x); P=np.abs(F)**2
    if x.ndim==3:
        nz,ny,nx=x.shape; dx,dy,dz=spc
        kx=2*np.pi*fftfreq(nx,dx); ky=2*np.pi*fftfreq(ny,dy); kz=2*np.pi*fftfreq(nz,dz)
        KZ,KY,KX=np.meshgrid(kz,ky,kx,indexing="ij"); K=np.sqrt(KX**2+KY**2+KZ**2); N=nx*ny*nz
    else:
        ny,nx=x.shape; dx,dy,_=spc
        kx=2*np.pi*fftfreq(nx,dx); ky=2*np.pi*fftfreq(ny,dy)
        KY,KX=np.meshgrid(ky,kx,indexing="ij"); K=np.sqrt(KX**2+KY**2); N=nx*ny
    K=fftshift(K); P=fftshift(P)/(N**2)
    q=K.ravel(); k=P.ravel()
    qpos=q[np.isfinite(q)&(q>0)]
    qmin=qpos.min(); qmax=qpos.max()
    bins=np.logspace(np.log10(qmin),np.log10(qmax),n+1)
    idx=np.digitize(q,bins)-1
    good=np.isfinite(k)&(idx>=0)&(idx<n)
    ss=np.bincount(idx[good],weights=k[good],minlength=n)
    cnt=np.bincount(idx[good],minlength=n).astype(float)
    qc=0.5*(bins[1:]+bins[:-1]); dq=np.diff(bins)
    with np.errstate(invalid="ignore",divide="ignore"):
        Ek=ss/dq
    m=np.isfinite(Ek)
    s=(x.var()/np.trapz(Ek[m],qc[m])) if np.any(m) and np.trapz(Ek[m],qc[m])>0 else 1.0
    return qc,Ek*s,x.var()

def slope_line(ax, k, E, slope, k1, k2, **kw):
    m = (k>0)&(E>0)&np.isfinite(k)&np.isfinite(E)&(k>=k1)&(k<=k2)
    if not np.any(m): return
    kk, ee = k[m], E[m]
    x = np.log(kk); y = np.log(ee)
    x0 = x.mean(); k0 = np.exp(x0)
    c  = np.mean(y - slope*(x - x0))
    A  = np.exp(c)
    ax.loglog(kk, A*(kk/k0)**slope, **kw)

def best_slope_line(ax, k, E, k1, k2, mode="syn", **kw):
    m = (k>0)&(E>0)&np.isfinite(k)&np.isfinite(E)&(k>=k1)&(k<=k2)
    if not np.any(m): return None
    kk, ee = k[m], E[m]
    x = np.log(kk); y = np.log(ee)
    x0 = x.mean(); k0 = np.exp(x0)
    dx = x - x0
    s  = np.sum((y - y.mean())*dx) / np.sum(dx*dx)
    c  = np.mean(y - s*dx)
    A  = np.exp(c)
    ax.loglog(kk, A*(kk/k0)**s, label=f"{mode} ({s:.2f})", **kw)
    return float(s)

def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    x,spc=load_field("mhd_fields.h5",DSET); kA,EA,varA=spectrum(x,spc,NBINS)
    x,spc=load_field("two_slope_2D_s4_r00.h5",DSET); kS,ES,varS=spectrum(x,spc,NBINS)

    plt.figure(figsize=(7,5))
    plt.loglog(kA,EA,label=f"Athena (var={varA:.3e})",color="lightblue")
    plt.loglog(kS,ES,label=f"Synthetic (var={varS:.3e})",color="salmon")

    k1S,k2S=np.nanmin(kS[kS>0]),np.nanmax(kS); ks=np.sqrt(k1S*k2S)

    ks*=1.5
    k2S*=0.75
    k1S =20/256

    print(k1S, ks)
    
    slope_line(plt,kS,ES,+1.5, k1S, ks, color="orangered", ls=":", label="syn +3/2")
    slope_line(plt,kS,ES,-5/3, ks, k2S, color="crimson",   ls=":", label="syn -5/3")
    sS1=best_slope_line(plt,kS,ES, k1S, ks, "syn",   color="firebrick",   ls="--")
    sS2=best_slope_line(plt,kS,ES, ks, k2S, "syn", color="darkred",    ls="--")    
    plt.axvline(ks, color="gray", ls=":", label=f"k={ks:.2f}", lw=1.2, alpha=0.8)

    k1A,k2A=np.nanmin(kA[kA>0]),np.nanmax(kA); ka=np.sqrt(k1A*k2A)
    ka*=1.5
    k2A*=0.6
    k1A =20/256
    print(k1A, ka)

    # slope_line(plt,kA,EA,-0.5, k1A, ka, color="dodgerblue", ls=":")
    # slope_line(plt,kA,EA,-1.5, ka,  k2A, color="royalblue",  ls=":")
    sA1=best_slope_line(plt,kA,EA, k1A, ka, "ath",   color="dodgerblue",        ls="--")
    sA2=best_slope_line(plt,kA,EA, ka,  k2A, "ath",  color="royalblue", ls="--")
    plt.axvline(ka, color="purple", ls=":", label=f"k={ka:.2f}", lw=1.2, alpha=0.8)

    if sS1 is not None or sS2 is not None or sA1 is not None or sA2 is not None:
        print("Best-fit slopes:")
        if sS1 is not None: print(f"  Synthetic low  interval: {sS1:.3f}")
        if sS2 is not None: print(f"  Synthetic high interval: {sS2:.3f}")
        if sA1 is not None: print(f"  Athena    low  interval: {sA1:.3f}")
        if sA2 is not None: print(f"  Athena    high interval: {sA2:.3f}")

    plt.xlabel(r"$k$"); plt.ylabel(r"$E(k)$"); plt.grid(True,which="both",alpha=.3)
    plt.legend(frameon=False,ncol=2); plt.xlim(20/256)
    plt.title(os.path.basename(H5)); plt.tight_layout()
    plt.savefig(OUT+".png",dpi=DPI); plt.savefig(OUT+".pdf"); plt.show()
    print("Saved →",os.path.abspath(OUT)+".png/.pdf")

if __name__=="__main__": main()
