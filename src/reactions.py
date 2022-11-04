#import gas as phase
import numpy as np

def dG_RT_F_H2O(T, H2, O2, H2O):
    # H2O formation reaction:
    # H2 + 1/2O2 -> H2O
    return (H2O.get_gibbs_RT(T) - H2.get_gibbs_RT(T) 
            - (1/2)*O2.get_gibbs_RT(T))

def dG_RT_F_OH(T, H2, O2, OH):
    # OH formation reaction:
    # 1/2H2 + 1/2O2 -> OH
    return (OH.get_gibbs_RT(T) - (1/2)*H2.get_gibbs_RT(T) 
            - (1/2)*O2.get_gibbs_RT(T))

def dG_RT_F_O(T, O2, O):
    # O formation reaction:
    # 1/2O2 -> O
    return (O.get_gibbs_RT(T) - (1/2)*O2.get_gibbs_RT(T))

def dG_RT_F_H(T, H2, H):
    # H formation reaction:
    # 1/2H2 -> H
    return (H.get_gibbs_RT(T) - (1/2)*H2.get_gibbs_RT(T))

def dG_RT_F_NO(T, N2, O2, NO):
    # NO formation reaction:
    # 1/2N2 + 1/2O2 -> NO
    return (NO.get_gibbs_RT(T) - (1/2)*N2.get_gibbs_RT(T) 
            - (1/2)*O2.get_gibbs_RT(T))

def kp(delta_G_RT_F):
    print('KP', np.exp(-delta_G_RT_F))
    return np.exp(-delta_G_RT_F)

#if __name__ == '__main__':
#
#    # State properties
#    p_ct = 20 * ct.one_atm
#    T_k_old = 2500 
#    
#    species = ['H2', 'O2', 'N2', 'H2O', 'OH', 'O', 'H', 'NO']
#    #species = 'H2'
#
#    # Define the vector of mole fractions X as:
#    # X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
#    X_k_old = (1/8)*np.ones(8)
#
#    # Define equivalence ration interval 
#    N = int((1.3-0.7)/0.1 + 1)
#    eq_ratio = np.linspace(0.7, 1.3, num=N)
#    
#    # Iterative method tolerance
#    error_norm = 1e-5
#
#    # general information
#    for specie in species[:]:
#        tmp = phase.gas(specie) 
#        #tmp.print_coefs()
#        #print('Enthalpy for ', specie, ':\n',  tmp.get_enthalpy_RT(T_k_old))
#        #print('Entropy for ', specie, ':\n',  tmp.get_entropy_R(T_k_old))
#        #print('Gibbs for ', specie, ':\n',  tmp.get_gibbs_RT(T_k_old))
#
#    # Initializing products:
#    H2 = phase.gas('H2')
#    O2 = phase.gas('O2')
#    N2 = phase.gas('N2')
#    H2O = phase.gas('H2O')
#    OH = phase.gas('OH')
#    O = phase.gas('O')
#    H = phase.gas('H')
#    NO = phase.gas('NO')
#
#    # H2O formation reaction:
#    # H2 + 1/2O2 -> H2O
#    dG_RT_F_H2O = (H2O.get_gibbs_RT(T_k_old) - H2.get_gibbs_RT(T_k_old) 
#                   - (1/2)*O2.get_gibbs_RT(T_k_old))
#    print('H2O')
#    kp_F_H2O = kp(dG_RT_F_H2O)
#
#    # OH formation reaction:
#    # 1/2H2 + 1/2O2 -> OH
#    dG_RT_F_OH = (OH.get_gibbs_RT(T_k_old) - (1/2)*H2.get_gibbs_RT(T_k_old) 
#                  - (1/2)*O2.get_gibbs_RT(T_k_old))
#    print('OH')
#    kp_F_OH = kp(dG_RT_F_OH)
#
#    # O formation reaction:
#    # 1/2O2 -> O
#    dG_RT_F_O = (O.get_gibbs_RT(T_k_old) - (1/2)*O2.get_gibbs_RT(T_k_old))
#    print('O')
#    kp_F_O = kp(dG_RT_F_O)
#    
#    # H formation reaction:
#    # 1/2H2 -> H
#    dG_RT_F_H = (H.get_gibbs_RT(T_k_old) - (1/2)*H2.get_gibbs_RT(T_k_old))
#    print('H')
#    kp_F_H = kp(dG_RT_F_H)
#
#    # NO formation reaction:
#    # 1/2N2 + 1/2O2 -> NO
#    dG_RT_F_NO = (NO.get_gibbs_RT(T_k_old) - (1/2)*N2.get_gibbs_RT(T_k_old) 
#                  - (1/2)*O2.get_gibbs_RT(T_k_old))
#    print('NO')
#    kp_F_NO = kp(dG_RT_F_NO)
