import nasapol as nasa
import reactions as rt 
import gas as phase 
import cantera as ct
import numpy as np

if __name__ == '__main__':

    # State properties
    p_ct = 20 * ct.one_atm
    T_k_old = 2500
    
    species = ['H2', 'O2', 'N2', 'H2O', 'OH', 'O', 'H', 'NO']
    #species = 'H2'

    # Initializing products:
    H2 = phase.gas('H2')
    O2 = phase.gas('O2')
    N2 = phase.gas('N2')
    H2O = phase.gas('H2O')
    OH = phase.gas('OH')
    O = phase.gas('O')
    H = phase.gas('H')
    NO = phase.gas('NO')

    # Define the vector of mole fractions X as:
    # X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    X_k_old = 0.1*np.ones(8)
    X_k = np.zeros(8)

    # Define equivalence ration interval 
    N = int((1.3-0.7)/0.1 + 1)
    eq_ratio = np.linspace(0.7, 1.3, num=N)
    
    # Iterative method tolerance
    error_norm = 1e-5

    #for specie in species:
    #    tmp = phase.gas(specie) 
    #    tmp.print_coefs()
    #    print('Enthalpy for ', specie, ':\n',  tmp.get_enthalpy_RT(T_k_old))
    #    print('Entropy for ', specie, ':\n',  tmp.get_entropy_R(T_k_old))
    #    print('Gibbs for ', specie, ':\n',  tmp.get_gibbs_RT(T_k_old))

    # Reaction values:
    kp_F_H2O = rt.kp(rt.dG_RT_F_H2O(T_k_old, H2, O2, H2O))
    kp_F_OH = rt.kp(rt.dG_RT_F_OH(T_k_old, H2, O2, OH))
    kp_F_O = rt.kp(rt.dG_RT_F_O(T_k_old, O2, O))
    kp_F_H = rt.kp(rt.dG_RT_F_H(T_k_old, H2, H))
    kp_F_NO = rt.kp(rt.dG_RT_F_NO(T_k_old, N2, O2, NO))

    for i in range(5):
        # Fixed point iterative method
        # From kp_F equations
        # X_H2O = kp_F_H2O*X_H2*\sqrt(X_O2)*\sqrt(p)
        X_k[3] = kp_F_H2O*X_k_old[0]*np.sqrt(X_k_old[1])*np.sqrt(p_ct)
        # X_OH = kp_F_OH*\sqrt{X_H2}*\sqrt{X_O2}
        X_k[4] = kp_F_OH*np.sqrt(X_k_old[0])*np.sqrt(X_k_old[1])
        # X_O = kp_F_O*\sqrt{X_O2}*\frac{1}{\sqrt{p}}
        X_k[5] = kp_F_O*np.sqrt(X_k_old[1])/np.sqrt(p_ct)
        # X_H = kp_F_H*\sqrt{X_H2}*\frac{1}{\sqrt{p}}
        X_k[6] = kp_F_H*np.sqrt(X_k_old[0])/np.sqrt(p_ct)
        # X_NO = kp_F_NO*\sqrt{X_H2}*\sqrt{X_O2}
        X_k[7] = kp_F_NO*np.sqrt(X_k_old[2])*np.sqrt(X_k_old[1])
        # From conservation of atoms
        # X_H2 = [2*\Phi*(2*X_O2 + X_H2O + X_OH + X_O + X_H) 
        #        -2*X_H2O - X_OH - X_H]/2
        X_k[0] = (1/2)*(2*eq_ratio[3]*(2*X_k_old[1] + X_k[3] + X_k[4] 
                        + X_k[5] + X_k[7]) - 2*X_k[3] - X_k[4] - X_k[6])
        # X_N2 = [3.76*(2*X_O2 + X_H2O + X_OH + X_O + X_H) - X_NO]/2
        X_k[2] = (1/2)*(3.76*(2*X_k_old[1] + X_k[3] + X_k[4] 
                        + X_k[5] + X_k[7]) - X_k[7])
        # X_O2 = 1 - \Sigma{X_i} # except for O2
        X_k[1] = 1 - X_k[0] - np.sum(X_k[2:])

        print('X', X_k)
        X_k_old = X_k

