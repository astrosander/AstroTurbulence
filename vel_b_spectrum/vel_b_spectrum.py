import numpy as np, h5py, matplotlib.pyplot as plt, os

def vector_spectrum(vx,vy,vz):
    Fvx=np.fft.fftn(vx); Fvy=np.fft.fftn(vy); Fvz=np.fft.fftn(vz)
    P=(np.abs(Fvx)**2+np.abs(Fvy)**2+np.abs(Fvz)**2)/vx.size
    nz,ny,nx=vx.shape
    kx=np.fft.fftfreq(nx)*nx; ky=np.fft.fftfreq(ny)*ny; kz=np.fft.fftfreq(nz)*nz
    KX,KY,KZ=np.meshgrid(kx,ky,kz,indexing="xy")
    Km=np.sqrt(KX**2+KY**2+KZ**2)
    ind=np.rint(Km).astype(int)
    kmax=int(ind.max())
    Ek=np.zeros(kmax+1); cnt=np.zeros(kmax+1,dtype=int)
    for i in range(1,kmax+1):
        m=(ind==i)
        if m.any():
            Ek[i]=P[m].sum()   # shell SUM -> E(k) ~ k^2 P3D(k)
            cnt[i]=m.sum()
    k=np.arange(kmax+1,dtype=float); m=(k>0)&(cnt>0)
    return k[m],Ek[m]

def slope_line(k,Ek,s=-5/3):
    m=(k>k.min()*2)&(k<k.max()*0.4)
    if not np.any(m): m=slice(len(k)//3,len(k)//2)
    kref=np.median(k[m]); Eref=np.interp(kref,k,Ek)
    A=Eref/(kref**s); return A*(k**s)

h5_path=r"..\faradays_angles_stats\lp_structure_tests\mhd_fields.h5"
with h5py.File(h5_path,"r") as f:
    vx=np.array(f["velx"],dtype=np.float32)
    vy=np.array(f["vely"],dtype=np.float32)
    vz=np.array(f["velz"],dtype=np.float32)
    bx=np.array(f["bcc1"],dtype=np.float32)
    by=np.array(f["bcc2"],dtype=np.float32)
    bz=np.array(f["bcc3"],dtype=np.float32)

    # vx = np.array(f["i_velocity"], dtype=np.float32)
    # vy = np.array(f["j_velocity"], dtype=np.float32)
    # vz = np.array(f["k_velocity"], dtype=np.float32)

    # bx = np.array(f["i_mag_field"], dtype=np.float32)
    # by = np.array(f["j_mag_field"], dtype=np.float32)
    # bz = np.array(f["k_mag_field"], dtype=np.float32)


k_v,Ev=vector_spectrum(vx,vy,vz)
k_b,Eb=vector_spectrum(bx,by,bz)
guide_v=slope_line(k_v,Ev,-5/3)
guide_b=slope_line(k_b,Eb,-5/3)

os.makedirs("fig_2024",exist_ok=True)
plt.figure(figsize=(10,4))
ax1=plt.subplot(1,2,1)
ax1.loglog(k_v,Ev,lw=1.8,label=r"$M_A=2.0$")
ax1.loglog(k_v,guide_v,"k--",lw=1.5,label=r"slope $-5/3$")
ax1.set_xlabel(r"$k\,L_{\rm box}/(2\pi)$"); ax1.set_ylabel(r"$E_v(k)$"); ax1.set_title("Velocity Spectrum"); ax1.legend(loc="lower left",fontsize=9)
ax2=plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
ax2.loglog(k_b,Eb,lw=1.8)
ax2.loglog(k_b,guide_b,"k--",lw=1.5)
ax2.set_xlabel(r"$k\,L_{\rm box}/(2\pi)$"); ax2.set_ylabel(r"$E_B(k)$"); ax2.set_title("Magnetic Field Spectrum")
plt.xlim(0,200)
plt.ylim(0.1, 10**6)

plt.tight_layout()
plt.savefig("fig_2024/spectrum.pdf",dpi=300)
plt.close()
