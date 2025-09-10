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

def slope_line(ax,k,E,slope,k1,k2,**kw):
    m=(k>=k1)&(k<=k2)&np.isfinite(E)
    if not np.any(m): return
    kk=k[m]; ee=E[m]; i=len(kk)//2; kr,Er=kk[i],ee[i]
    ax.loglog(kk,Er*(kk/kr)**slope,**kw)

def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    x,spc=load_field("mhd_fields.h5",DSET); kA,EA,varA=spectrum(x,spc,NBINS)
    x,spc=load_field("two_slope_2D_s4_r00.h5",DSET); kS,ES,varS=spectrum(x,spc,NBINS)

    fig=plt.figure(figsize=(7,5))
    plt.loglog(kA,EA,label=f"Athena (var={varA:.3e})",color="lightblue")
    plt.loglog(kS,ES,label=f"Synthetic (var={varS:.3e})",color="salmon")

    k1S,k2S=np.nanmin(kS[kS>0]),np.nanmax(kS)
    ks=np.sqrt(k1S*k2S)
    slope_line(plt,g:=kS,ES, +1.5, k1S,ks*1.5, color="orangered", ls=":", label="syn +3/2")
    slope_line(plt,g,ES, -5/3, ks*1.5, k2S*0.75, color="crimson", ls=":", label="syn -5/3")

    k1A,k2A=np.nanmin(kA[kA>0]),np.nanmax(kA)
    ka=np.sqrt(k1A*k2A)
    slope_line(plt,kA,EA, -0.5, k1A,ka, color="dodgerblue", ls=":", label="ath -1/2")
    slope_line(plt,kA,EA, -1.5, ka, k2A, color="royalblue", ls=":", label="ath -3/2")

    plt.xlabel(r"$k$"); plt.ylabel(r"$E(k)$"); plt.grid(True,which="both",alpha=.3)
    plt.legend(frameon=False,ncol=2); plt.xlim(20/256)
    plt.title(os.path.basename(H5)); plt.tight_layout()
    plt.savefig(OUT+".png",dpi=DPI); plt.savefig(OUT+".pdf"); plt.show()
    print("Saved →",os.path.abspath(OUT)+".png/.pdf")
if __name__=="__main__": main()
