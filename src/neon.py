import nasapol as nasa
import reactions as rt 
import gas as phase 
import num_method as num
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../latex.mplstyle')

def molar_frac(T, p, eq_ratio, H2, O2, N2, H2O, OH, O, H, NO, Ne, n_Ne):

    # Reaction values:
    dG_F_H2O = rt.dG_RT_F_H2O(T, H2, O2, H2O)
    #print(f"Kp_F H2O = {kp_F_H2O}")
    dG_F_OH = rt.dG_RT_F_OH(T, H2, O2, OH)
    #print(f"Kp_F OH = {kp_F_OH}")
    dG_F_O = rt.dG_RT_F_O(T, O2, O)
    #print(f"Kp_F O = {kp_F_O}")
    dG_F_H = rt.dG_RT_F_H(T, H2, H)
    #print(f"Kp_F H = {kp_F_H}")
    dG_F_NO = rt.dG_RT_F_NO(T, N2, O2, NO)
    #print(f"Kp_F NO = {kp_F_NO}")

    dG_F = np.array([dG_F_H2O, dG_F_OH, dG_F_O, dG_F_H, dG_F_NO])

    # Iterative method tolerance
    max_it_b = 15
    error_norm = 0.01
    error_T = 0.01 

    # Define the vector of mole fractions X as:
    # X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO, X_Ne]
    X_guess = np.array([3e-3, 1e-3, 0.45, 0.23, 1e-3, 1e-5, 1e-4, 7e-4, 0.3])
    n_tot_f = rt.get_final_moles(X_guess, eq_ratio)
    X_guess[8] = n_Ne/n_tot_f

    X_k_old = X_guess
    T_k_old = T 
    error_old = 1
    #n_Ne_old = 2e-2

    for i in range(10):

        #n_tot_i = rt.get_initial_moles_ne(eq_ratio, n_Ne)
        n_tot_f = rt.get_final_moles(X_guess, eq_ratio)
        #n_Ne = n_tot_f*X_k_old[8]
        #print(f"n_Ne: {n_Ne:.2f} moles")
        H_react = rt.H_react_Ne(eq_ratio, H2, O2, N2, n_Ne, Ne)
        #print(f"H_react: {H_react:.2f} J")
        H_prod = rt.H_prod_Ne(T, X_guess, eq_ratio, H2, O2, N2, 
                              H2O, OH, O, H, NO, Ne, n_tot_f)
        #print(f"H_prod: {H_prod:.2f} J")

        # Euler's method for:
        # x_k = x_k-1 - J^-1(x_k-1)F(x_k-1), 
        # given J as the jacobian of the F function.
        j_inv = np.linalg.inv(num.jacobian_Ne(X_k_old, p, T, n_tot_f,
                                              H2, O2, N2, H2O, OH, O, H, NO, Ne))
        F_X_k_old = num.F_Ne(X_k_old, p, eq_ratio, dG_F, H_react, H_prod)
        j_inv_F_X_k_old = np.matmul(j_inv, F_X_k_old).T
        j_inv_F_X_k_old = np.squeeze(j_inv_F_X_k_old)
        #print("J^-1F = ", j_inv_F_X_k_old)
        #print(j_inv_F_X_k_old.shape)
        X_k = np.add(X_k_old, -j_inv_F_X_k_old)

        # Verifying the validity of the results
        #print(f"Species = {species}")
        #print(f"X = {X_k}")
        #print(f"Sum of X = {np.sum(X_k)}")
        #print(f"Error[{i+1}] = {np.linalg.norm(X_k - X_k_old)}")

        #print('T = ', T_k_old)
        error = np.linalg.norm(X_k - X_k_old)

        if (error > error_old) or np.isnan(error):
        #if np.isnan(error):
            print(f"Euler method diverging at iteration {i+1}")
            X_k = X_k_old # returning to the old value
            T_k = num.bisection_Ne(T_k_old, H_react, X_k, eq_ratio, H2, O2, N2, H2O, 
                                OH, O, H, NO, Ne, n_tot_f, error_T, max_it_b, i+1)
            break
        else:
            error_old = error
            X_k_old = X_k
            T_k = num.bisection_Ne(T_k_old, H_react, X_k, eq_ratio, H2, O2, N2, H2O, 
                                OH, O, H, NO, Ne, n_tot_f, error_T, max_it_b, i+1)
            T_k_old = T_k

    return X_k, T_k

if __name__ == '__main__':

    # State properties
    p = 20
    eq_ratio = 1
    T_target = 2268.17
    T = 2500 

    # Initializing products:
    species = ['H2', 'O2', 'N2', 'H2O', 'OH', 'O', 'H', 'NO', 'Ne']
    H2 = phase.gas('H2')
    O2 = phase.gas('O2')
    N2 = phase.gas('N2')
    H2O = phase.gas('H2O')
    OH = phase.gas('OH')
    O = phase.gas('O')
    H = phase.gas('H')
    NO = phase.gas('NO')
    Ne = phase.gas('Ne')

    n_Ne_a = np.linspace(0, 0.01, 10)

    for n_Ne in n_Ne_a:

        X, T = molar_frac(T, p, eq_ratio, H2, O2, N2, H2O, OH, O, H, NO, Ne, n_Ne)
        
        print(f"n_Ne = {n_Ne:.5f}")
        print(f"X = {X}")
        print(f"T = {T}")

        if T<T_target:
            # Properties:
            n_tot_f = rt.get_final_moles(X, eq_ratio)
            H_react = rt.H_react_Ne(eq_ratio, H2, O2, N2, n_Ne, Ne)
            H_prod = rt.H_prod_Ne(T, X, eq_ratio, H2, O2, N2, 
                                  H2O, OH, O, H, NO, Ne, n_tot_f)

            print(f"---End results for n_Ne of {n_Ne:.5f}")
            print(f"n_tot = {n_tot_f:.2f}")
            print(f"H_react = {H_react:.2f} J")
            print(f"H_prod({T:.2f}) = {H_prod:.2f} J")
            print(f"Flame temperature = {T:.2f} K")
            print(f"X_H2 = {X[0]*100:.2f}%")
            print(f"X_O2 = {X[1]*100:.2f}%")
            print(f"X_N2 = {X[2]*100:.2f}%")
            print(f"X_H2O = {X[3]*100:.2f}%")
            print(f"X_OH = {X[4]*100:.2f}%")
            print(f"X_O = {X[5]*100:.2f}%")
            print(f"X_H = {X[6]*100:.2f}%")
            print(f"X_NO = {X[7]*100:.2f}%")
            print(f"X_Ne = {X[8]*100:.2f}%")
            break
