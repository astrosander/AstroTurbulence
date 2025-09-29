import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.ticker as mticker

def fbm2d(ny,nx,alpha,seed=0):
    rng=np.random.default_rng(seed)
    kx=np.fft.fftfreq(nx); ky=np.fft.fftfreq(ny)
    KX,KY=np.meshgrid(kx,ky,indexing="xy")
    k=np.sqrt(KX**2+KY**2); k[0,0]=1.0
    amp=1.0/(k**(alpha/2.0))
    phase=np.exp(1j*2*np.pi*rng.random((ny,nx)))
    f=np.fft.ifft2(amp*phase).real
    f-=f.mean(); f/=f.std()+1e-12
    return f

def directional_spectrum(P):
    Q=np.real(P); U=np.imag(P)
    A=np.sqrt(Q**2+U**2)+1e-30
    c2=Q/A; s2=U/A
    F1=np.fft.fftshift(np.fft.fft2(c2))
    F2=np.fft.fftshift(np.fft.fft2(s2))
    P2=(np.abs(F1)**2+np.abs(F2)**2)/P.size
    ny,nx=P2.shape
    ky=np.fft.fftshift(np.fft.fftfreq(ny)); kx=np.fft.fftshift(np.fft.fftfreq(nx))
    KX,KY=np.meshgrid(kx,ky)
    kr=np.sqrt(KX**2+KY**2).ravel(); p=P2.ravel()
    idx=np.argsort(kr); kr=kr[idx]; p=p[idx]
    n=max(24,int(np.sqrt(nx*ny)))
    edges=np.linspace(kr.min(),kr.max(),n+1)
    centers=0.5*(edges[1:]+edges[:-1])
    Pk=np.zeros_like(centers); cnt=np.zeros_like(centers)
    ind=np.digitize(kr,edges)-1
    v=(ind>=0)&(ind<n)
    for i,val in zip(ind[v],p[v]): Pk[i]+=val; cnt[i]+=1
    cnt[cnt==0]=1; Pk/=cnt
    return centers,Pk

def scalar_psd(field2d):
    F=np.fft.fftshift(np.fft.fft2(field2d))
    P2=(np.abs(F)**2)/field2d.size
    ny,nx=P2.shape
    ky=np.fft.fftshift(np.fft.fftfreq(ny)); kx=np.fft.fftshift(np.fft.fftfreq(nx))
    KX,KY=np.meshgrid(kx,ky)
    kr=np.sqrt(KX**2+KY**2).ravel(); p=P2.ravel()
    idx=np.argsort(kr); kr=kr[idx]; p=p[idx]
    n=max(24,int(np.sqrt(nx*ny)))
    edges=np.linspace(kr.min(),kr.max(),n+1)
    centers=0.5*(edges[1:]+edges[:-1])
    Pk=np.zeros_like(centers); cnt=np.zeros_like(centers)
    ind=np.digitize(kr,edges)-1
    v=(ind>=0)&(ind<n)
    for i,val in zip(ind[v],p[v]): Pk[i]+=val; cnt[i]+=1
    cnt[cnt==0]=1; Pk/=cnt
    return centers,Pk

ny=nx=512
alpha_emit=1.8
alpha_rm=1.30
psi_emit=fbm2d(ny,nx,alpha_emit,seed=1)*0.20
RM=fbm2d(ny,nx,alpha_rm,seed=2)
#lams=np.logspace(-2,0.6,16)

base_lams = [0.01, 0.1, 0.3, 0.6, 2.0, 4.0]

# focus on transition regime with 4 values
transition = np.linspace(0.97, 1.3, 4)

# base_lams = []
# transition = np.linspace(0.1, 1, 10)

# merge and sort
show_lams = sorted(base_lams + list(transition))

lams=np.linspace(0.14, 1.0, 100)

print(show_lams)
curves=[]
for lam in show_lams:
    ang=psi_emit+(lam**2)*RM
    P=np.exp(2j*ang)
    k,Pk=directional_spectrum(P)
    m=(k>0)&(Pk>0)
    curves.append((k[m],Pk[m],lam))

kx_list=[]
for lam in lams:
    k1,P1=scalar_psd(psi_emit)
    k2,P2=scalar_psd((lam**2)*RM)
    P2i=np.interp(k1,k2,P2,left=P2[0],right=P2[-1])
    j=np.argmin(np.abs(np.log10(P1+1e-30)-np.log10(P2i+1e-30)))
    kx_list.append(k1[j])

os.makedirs("fig_2024",exist_ok=True)
plt.figure(figsize=(10,4.2))
ax1=plt.subplot(1,2,1)
for k,Pk,lam in curves:
    ax1.loglog(k,Pk,label=fr"$\lambda={lam:.2f}$")
ax1.set_xlabel(r"$k$"); ax1.set_ylabel(r"$P_{\rm dir}(k)$")
# ax1.set_title("Emission + Faraday Screen")
ax1.legend(fontsize=8)

ax2=plt.subplot(1,2,2)
ax2.loglog(lams,kx_list, label="Measured crossover")#,"o-"

# x=np.array(lams); y0=kx_list[10]*(x[10]**2)/(x**5)


def fit_psd(k, Pk, kmin_frac=0.05, kmax_frac=0.4):
    m = (k>k.min()/k.min()*k.min()) & (k>k.max()*kmin_frac) & (k<k.max()*kmax_frac) & (Pk>0)
    if m.sum()<8:
        m = slice(len(k)//6, len(k)//2)
    x = np.log10(k[m]); y = np.log10(Pk[m])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]  # y = slope*x + intercept
    gamma = -slope
    A0 = 10**intercept
    return gamma, A0

k_psi, P_psi = scalar_psd(psi_emit)
k_rm,  P_rm  = scalar_psd(RM)
# fit power-laws: P ~ A k^{-exponent}
delta, Apsi = fit_psd(k_psi, P_psi)
gamma, ARM  = fit_psd(k_rm,  P_rm)


def kx_pred(lam):
    expo = 1.3-1.8
    return ( (lam**4) )**(1.0/expo)

x=np.array(lams); y0=kx_pred(x)

ax2.loglog(x[x>0.3], y0[x>0.3]/8000, "k--", lw=1, label= r"$k_\times \propto \lambda^{4/(\gamma-\delta)}$")
ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel(r"$k_\times(\lambda)$")

# ax2.set_title(r"Cross-over $k_\times$: $\psi_{\rm emit}$ vs $\lambda^2{\rm RM}$")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_2024/transition_lambda.pdf",dpi=300)
plt.close()
