#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_psa_results(csv_file, json_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    k = data[:, 0]
    Pk = data[:, 1]
    
    with open(json_file, 'r') as f:
        fit_results = json.load(f)
    
    return k, Pk, fit_results

def plot_psa_results(k, Pk, fit_results, output_file="psa_plot.pdf"):
    plt.figure(figsize=(10, 8))
    
    plt.loglog(k[k > 0], Pk[k > 0], 
               marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    
    k_min, k_max = 1.0, 8.0
    k_min_ref = 1.0 / (k_max**2)
    k_max_ref = 1.0 / (k_min**2)
    
    if k_min_ref > 0 and k_max_ref > 0:
        k_ref = np.logspace(np.log10(k_min_ref), 
                            np.log10(k_max_ref), 100)

        scale_factor = k[k > 0].min() * 250

        Pk_ref = scale_factor * k_ref**(-11/3)

        print(Pk[-1])
        
        plt.loglog(k_ref, Pk_ref, ':', linewidth=2, 
                  label='-11/3', color='green')
    
    plt.xlabel('k', fontsize=12)
    plt.ylabel('P(k)', fontsize=14)
    plt.title('PSA: Spatial Power Spectrum (Syn)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    return plt.gcf()

def main():
    csv_file = "COMP_dirPk.csv"
    json_file = "PSA_out_fit.json"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run psa_simulation.py first.")
        return
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Run psa_simulation.py first.")
        return
    
    k, Pk, fit_results = load_psa_results(csv_file, json_file)
    
    print("PSA Simulation Results Summary:")
    print(f"  Number of k points: {len(k)}")
    print(f"  k range: {k.min():.4f} to {k.max():.4f}")
    print(f"  P(k) range: {Pk.min():.2e} to {Pk.max():.2e}")
    print(f"  Fitted slope: {fit_results['slope']:.3f}")
    print(f"  R²: {fit_results['r2']:.3f}")
    print(f"  Fit range: k = {fit_results['k_min']:.3f} to {fit_results['k_max']:.3f}")
    
    plot_psa_results(k, Pk, fit_results)
    plt.show()

if __name__ == "__main__":
    main()
