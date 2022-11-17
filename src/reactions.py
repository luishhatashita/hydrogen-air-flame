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
    #print('Kp_F =', np.exp(-delta_G_RT_F))
    return np.exp(-delta_G_RT_F)

def get_initial_moles(phi):
    return phi + 0.5 + (3.76/2)

def H_react(phi, H2, O2, N2):
    n_tot_i = get_initial_moles(phi)

    H_react = n_tot_i*((phi/n_tot_i)*H2.get_enthalpy_per_mole(298) 
                      + (0.5/n_tot_i)*O2.get_enthalpy_per_mole(500) 
                      + ((3.76/2)/n_tot_i)*N2.get_enthalpy_per_mole(500))

    return H_react

def get_final_moles(X, phi):
    # Let X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]

    # from H_atoms = 2*phi = n_tot_f * (2*X_H2 + X_H20 + X_OH + X_H)
    n_tot_f_H = 2 * phi / (2*X[0] + 2*X[3] + X[4] + X[6])

    # from O_atoms = 1 = n_tot_f * (2*X_O2 + X_H20 + X_OH + X_O + X_NO)
    n_tot_f_O = 1 / (2*X[0] + X[3] + X[4] + X[5] + X[7])

    # from N_atoms = 3.76 = n_tot_f * (2*X_N2 + X_NO)
    n_tot_f_N = 3.76 / (2*X[2] + X[7])
    #print([n_tot_f_H, n_tot_f_O, n_tot_f_N])
    return np.average([n_tot_f_H, n_tot_f_O, n_tot_f_N])

def H_prod(T, X, phi, H2, O2, N2, H2O, OH, O, H, NO):
    n_tot_f = get_final_moles(X, phi)
    #print(n_tot_f)
    # Let X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    n_h_vect = n_tot_f * np.array(
                [H2.get_enthalpy_per_mole(T), O2.get_enthalpy_per_mole(T), 
                 N2.get_enthalpy_per_mole(T), H2O.get_enthalpy_per_mole(T),
                 OH.get_enthalpy_per_mole(T), O.get_enthalpy_per_mole(T), 
                 H.get_enthalpy_per_mole(T), NO.get_enthalpy_per_mole(T)]
                )

    #n_h_vect = [n_tot_f * h for h in h_vect]
    #print(n_h_vect)
    #print("H_prod: ", np.dot(X, n_h_vect))
    return np.dot(X, n_h_vect)
