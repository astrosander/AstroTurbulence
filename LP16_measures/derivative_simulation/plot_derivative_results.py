#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_derivative_results(csv_file, json_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    lambda2 = data[:, 0]
    var_dP = data[:, 1]
    
    with open(json_file, 'r') as f:
        fit_results = json.load(f)
    
    return lambda2, var_dP, fit_results

def plot_derivative_results(lambda2, var_dP, fit_results, output_file="derivative_plot.png"):
    plt.figure(figsize=(10, 8))
    plt.loglog(lambda2[lambda2 > 0], var_dP[lambda2 > 0], 
               marker='o', linestyle='none', markersize=6, 
               alpha=0.7, label='Data', color='blue')
    k_min, k_max = 1.0, 2.8
    lambda2_min_ref = 1.0 / (k_max**2)
    lambda2_max_ref = 1.0 / (k_min**2)
    if lambda2_min_ref > 0 and lambda2_max_ref > 0:
        lambda2_ref = np.logspace(np.log10(lambda2_min_ref), 
                                  np.log10(lambda2_max_ref), 100)

        scale_factor = lambda2[lambda2 > 0].min() * 120000000

        var_ref = scale_factor * lambda2_ref**(-5/3)

        print(var_dP[-1])
        
        plt.loglog(lambda2_ref, var_ref, ':', linewidth=2, 
                  label='-5/3', color='green')
    
    plt.xlabel(r'$\lambda^2$ (m$^2$)', fontsize=12)
    plt.ylabel(r'$\mathrm{Var}\!\left[\frac{\partial P}{\partial (\lambda^2)}\right]$', fontsize=14)
    plt.title('Derivative-based Polarization Statistic', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    return plt.gcf()

def main():
    csv_file = "DER_out_var_dP.csv"
    json_file = "DER_out_fit.json"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run derivative_simulation.py first.")
        return
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Run derivative_simulation.py first.")
        return
    
    lambda2, var_dP, fit_results = load_derivative_results(csv_file, json_file)
    print("Derivative Simulation Results Summary:")
    print(f"  Number of λ² points: {len(lambda2)}")
    print(f"  λ² range: {lambda2.min():.4f} to {lambda2.max():.4f}")
    print(f"  Var[dP/dλ²] range: {var_dP.min():.2e} to {var_dP.max():.2e}")
    print(f"  Fitted slope: {fit_results['slope']:.3f}")
    print(f"  R²: {fit_results['r2']:.3f}")
    
    plot_derivative_results(lambda2, var_dP, fit_results)
    plt.show()

if __name__ == "__main__":
    main()
