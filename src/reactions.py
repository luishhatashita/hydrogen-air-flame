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
