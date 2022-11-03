import nasapol as nasa
import cantera as ct
import numpy as np
import math

#for specie in species:
#    gas = ct.Solution('gri30.yaml')
#    X_str = specie + ':1'
#    gas.TPX = T_k_old, p_ct, X_str
#    print(gas())
#    print(gas.delta_standard_gibbs[0]/1e3)
#    gibbs_free = gas.standard_gibbs_RT[0] * (ct.gas_constant * T_k_old) / 1e3
#    print(gibbs_free)

def enthalpy_RT(T, coefs):
    h_RT = np.longdouble(coefs[0] + coefs[1]*T/2 + coefs[2]*(T**2)/3 + coefs[3]*(T**3)/4 
           + coefs[4]*(T**4)/5 + coefs[5]/T)

    return h_RT

def entropy_R(T, coefs):
    s_R = np.longdouble(coefs[0]*np.log(T) + coefs[1]*T + coefs[2]*(T**2)/2 + coefs[3]*(T**3)/3 
           + coefs[4]*(T**4/4) + coefs[5])

    return s_R 

def gibbs_RT(h_RT, s_R):
    return np.longdouble(h_RT - s_R)

def gibbs_RT_f(specie, T):
    if specie == 'H2O':
        G_RT_F = gibbs_RT(enthalpy_RT(T, specie)) - \
                 gibbs_RT(enthalpy_RT(T, 'H2'))
def kp(specie, T):
    if specie == 'H2O':
        
    return np.longdouble(np.exp(g_RT))

if __name__ == '__main__':

    # State properties
    p_ct = 20 * ct.one_atm
    T_k_old = 500 
    
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

    for specie in species[3:]:
        lines = nasa.find_element_lines(specie)
        coefs = nasa.nasa_7_coeff(T_k_old, lines)

        print('Calculating for: ', specie)

        # Calculating thermodynamic properties
        h_RT = enthalpy_RT(T_k_old, coefs)
        print('Enthalpy: ', h_RT)
        s_R = entropy_R(T_k_old, coefs)
        print('Entropy: ', s_R)
        g_RT = gibbs_RT(h_RT, s_R)
        print('Gibbs: ', g_RT)
        kp_value = kp(g_RT)
        print('kp: ', kp_value)
