import json, numpy as np, matplotlib.pyplot as plt

def _slope_guide(k, E, exponent=-5/3, f1=0.07, f2=0.35, s=10):
    klo, khi = int(len(k)*f1), int(len(k)*f2)
    kref = np.median(k[klo:khi]) if khi > klo else k[len(k)//3]
    Eref = np.interp(kref, k, E)
    return s * Eref / (kref**exponent) * k**exponent

def replot(npz="M_A=2.0, 792^3, t=1.0.npz", meta="M_A=2.0, 792^3, t=1.0.json", out="spectrum_replot.png"):
    data_files = [
        ("M_A=0.8, 256^3.npz", "M_A=0.8, 256^3.json", r"$M_A=0.8, 256^3$", "black", 1.0),
        ("M_A=2.0, 792^3, t=0.5.npz", "M_A=2.0, 792^3, t=0.5.json", r"$M_A=2.0, 792^3, t=0.5$", "lightblue", 0.4),
        ("M_A=2.0, 792^3, t=1.0.npz", "M_A=2.0, 792^3, t=1.0.json", r"$M_A=2.0, 792^3, t=1.0$", "darkblue", 1.0),
        ("M_A=2.0, 512^3, t=0.5.npz", "M_A=2.0, 512^3, t=0.5.json", r"$M_A=2.0, 512^3, t=0.5$", "red", 0.4),
        ("M_A=2.0, 512^3, t=1.0.npz", "M_A=2.0, 512^3, t=1.0.json", r"$M_A=2.0, 512^3, t=1.0$", "orange", 0.7),
        ("M_A=2.0, 512^3, t=1.5.npz", "M_A=2.0, 512^3, t=1.5.json", r"$M_A=2.0, 512^3, t=1.5$", "darkred", 1.0)
    ]
    
    plt.rcParams.update({"font.family":"STIXGeneral","font.size":12,"mathtext.fontset":"stix"})
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    
    # Load main data for slope guide
    a_main = np.load(npz); m_main = json.load(open(meta))
    kv_main, Ev_main, kb_main, Eb_main, kA_main = a_main["k_v"], a_main["E_v"], a_main["k_b"], a_main["E_b"], m_main["kA"]
    gv, gb = _slope_guide(kv_main, Ev_main), _slope_guide(kb_main, Eb_main)
    
    # Plot slope guide
    mask_main = kv_main <= 792/2
    for ax, g in [(ax1, gv[mask_main]), (ax2, gb[mask_main])]:
        ax.loglog(kv_main[mask_main], g*50, "--", lw=1.4, label="slope: $-5/3$", c='k')
    
    # Plot kA lines and data curves
    for npz_file, meta_file, label, color, alpha in data_files:
        try:
            a_add = np.load(npz_file)
            m_add = json.load(open(meta_file))
            kv_add, Ev_add, kb_add, Eb_add, kA_add = a_add["k_v"], a_add["E_v"], a_add["k_b"], a_add["E_b"], m_add["kA"]
            
            # Get resolution limit
            if "792" in label:
                xlim_max = 792/2
            elif "512" in label:
                xlim_max = 512/2
            elif "256" in label:
                xlim_max = 256/2
            else:
                xlim_max = 792/2  # default
            
            # Filter data
            mask = kv_add <= xlim_max
            kv_f, Ev_f, kb_f, Eb_f = kv_add[mask], Ev_add[mask], kb_add[mask], Eb_add[mask]
            
            # Plot kA line if within limit
            if kA_add <= xlim_max:
                ax1.axvline(kA_add, ls=":", lw=1.0, c=color, alpha=alpha*0.7)
                ax2.axvline(kA_add, ls=":", lw=1.0, c=color, alpha=alpha*0.7)
            
            # Plot spectra
            # if "256" in label:
            #     Ev_f *= 30 
            #     Eb_f *= 30

            ax1.loglog(kv_f, Ev_f, lw=1.5, label=label, c=color, alpha=alpha)
            ax2.loglog(kb_f, Eb_f, lw=1.5, label=label, c=color, alpha=alpha)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    # Add main kA line
    if kA_main <= 792/2:
        ax1.axvline(kA_main, ls="--", lw=1.2, c='r', label=r"$k_A$")
        ax2.axvline(kA_main, ls="--", lw=1.2, c='r', label=r"$k_A$")
    
    # Format axes
    for ax, lab in [(ax1, r"$E_v(k)$"), (ax2, r"$E_B(k)$")]:
        ax.set_xlabel(r"$k\frac{L_{\rm box}}{2\pi}$")
        ax.set_ylabel(lab)
        ax.set_ylim(1e-8,1)
        ax.set_xlim(1, 792/2)
        ax.legend(loc="lower left", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.show()
    plt.close()
    print("Recreated:", out)

if __name__ == "__main__":
    replot()
