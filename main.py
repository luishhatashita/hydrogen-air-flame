import reactions as rt 
import cantera as ct
import numpy as np
import sys

sys.path.insert(0, './src')

if __name__ == '__main__':

    # State properties
    p_ct = 20 * ct.one_atm
    T_k_old = 2500
    
    species = ['H2', 'O2', 'N2', 'H2O', 'OH', 'O', 'H', 'NO']
    #species = 'H2'

    # Define the vector of mole fractions X as:
    # X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    X_k_old = (1/8)*np.ones(8)

    # Define equivalence ration interval 
    N = int((1.3-0.7)/0.1 + 1)
    eq_ratio = np.linspace(0.7, 1.3, num=N)
    
    # Iterative method tolerance
    error_norm = 1e-5

    for specie in species:
        lines = src.find_element_lines(specie)
        coefs = src.nasa_7_coeff(T_k_old, lines)
        print('Calculating for: ', specie)
        h = enthalpy(T_k_old, coefs)
        print(h)
