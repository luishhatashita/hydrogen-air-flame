import nasapol as nasa
import reactions as rt 
import gas as phase 
import num_method as num
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../latex.mplstyle')

#def interpolate_x(eq_ratio):
#    df = pd.read_csv('../data/x_guess.csv')

def comp_temp(T_guess, X_guess, p, phi, H_react, H2, O2, N2, H2O, OH, O, H, NO):
    print(f"---Start: p = {p}, equivalence ratio = {phi:.1f}")
    # Iterative method tolerance
    max_it_b = 15
    error_norm = 0.01
    error_T = 0.01 

    X_k_old = X_guess
    T_k_old = T_guess 
    X_k = np.zeros(8)

    error_old = 1

    for i in range(10):
        #print(f"---Iteration {i+1} - Euler for X") 

        # Reaction values:
        kp_F_H2O = rt.kp(rt.dG_RT_F_H2O(T_k_old, H2, O2, H2O))
        #print(f"Kp_F H2O = {kp_F_H2O}")
        kp_F_OH = rt.kp(rt.dG_RT_F_OH(T_k_old, H2, O2, OH))
        #print(f"Kp_F OH = {kp_F_OH}")
        kp_F_O = rt.kp(rt.dG_RT_F_O(T_k_old, O2, O))
        #print(f"Kp_F O = {kp_F_O}")
        kp_F_H = rt.kp(rt.dG_RT_F_H(T_k_old, H2, H))
        #print(f"Kp_F H = {kp_F_H}")
        kp_F_NO = rt.kp(rt.dG_RT_F_NO(T_k_old, N2, O2, NO))
        #print(f"Kp_F NO = {kp_F_NO}")

        kp_vector = np.array([kp_F_H2O, kp_F_OH, kp_F_O, kp_F_H, kp_F_NO])

        # Euler's method for:
        # x_k = x_k-1 - J^-1(x_k-1)F(x_k-1), 
        # given J as the jacobian of the F function.
        j_inv = np.linalg.inv(num.jacobian(X_k_old, p))
        F_X_k_old = num.F(X_k_old, p, phi, kp_vector)
        j_inv_F_X_k_old = np.matmul(j_inv, F_X_k_old).T
        j_inv_F_X_k_old = np.squeeze(j_inv_F_X_k_old)
        #print("J^-1F = ", j_inv_F_X_k_old)
        #print(j_inv_F_X_k_old.shape)
        X_k = np.add(X_k_old, -j_inv_F_X_k_old)

        # Verifying the validity of the results
        #print(f"Species = {species}")
        #print(f"X = {X_k}")
        #print(f"Sum of X = {np.sum(X_k)}")
        print(f"Error[{i+1}] = {np.linalg.norm(X_k - X_k_old)}")

        #print('T = ', T_k_old)
        error = np.linalg.norm(X_k - X_k_old)

        #if (error > error_old) or np.isnan(error):
        if np.isnan(error):
            print(f"Euler method diverging at iteration {i+1}")
            X_k = X_k_old # returning to the old value
            T_k = num.bisection(T_k_old, H_react, X_k, phi, H2, O2, N2, 
                            H2O, OH, O, H, NO, error_T, max_it_b, i+1)
            break
        else:
            error_old = error
            X_k_old = X_k
            T_k = num.bisection(T_k_old, H_react, X_k, phi, H2, O2, N2, 
                                H2O, OH, O, H, NO, error_T, max_it_b, i+1)
            T_k_old = T_k

    return T_k_old, X_k_old

if __name__ == '__main__':

    # State properties
    ps = np.linspace(2, 10, 9) #* ct.one_atm
    
    # Define equivalence ration interval 
    N = int((1.3-0.7)/0.1 + 1)
    #N = 1
    eq_ratios = np.linspace(0.7, 1.3, num=N)
    #eq_ratios = [1] 
    
    T_guess_eq = [2200, 2200, 2200, 2500, 2500, 2500, 2500]
    #T_guess_eq = [2500]
    # Define the vector of mole fractions X as:
    # X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    X_guess_eq = np.array(
        [[2e-4, 5e-2, 0.65, 0.25, 2.4e-3, 1e-4, 1.7e-5, 5e-3],
         [8e-4, 3e-2, 0.67, 0.28, 4e-3, 2e-4, 7.7e-5, 6e-3],
         [1.3e-3, 1.4e-2, 0.7, 0.3, 5e-3, 3e-4, 2.7e-4, 5e-3],
         [1e-2, 5e-4, 0.64, 0.3, 5e-3, 2e-4, 8e-4, 2e-3],
         [3e-2, 1e-4, 0.62, 0.3, 3e-3, 8e-5, 1.5e-3, 1e-3],
         [6e-2, 5e-5, 0.6, 0.3, 2e-3, 3e-5, 1.7e-3, 3e-4],
         [1e-1, 1e-5, 0.6, 0.3, 1e-3, 1e-5, 1.7e-3, 1e-4]]
    )
    #print(f"Sum X_k_old: {np.sum(X_k_old)}")
    X_matrix_eq = np.zeros([N, 8])
    T_final_eq = np.zeros(N)
    X_matrix_p = np.zeros([len(ps), 8])
    T_final_p = np.zeros(len(ps))

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

    # Equivalence ratio study - \Phi \in [0.7, 1.3] at 20 atm
    for i, eq_ratio in enumerate(eq_ratios):

        # Reactants calculation of initial flame composition
        # Total enthalpy (ie extensive) = 
        # n_tot * (X_H2 * h_H2(T_H2) + X_O2 * h_O2(T_O2) +X_N2 * h_N2(T_N2)) 
        # where intensive enthalpy = H/RT * R_bar*T
        H_react = rt.H_react(eq_ratio, H2, O2, N2) 
        #print(f"H_react: {H_react:.2f} J")
        
        X_guess = np.array(X_guess_eq[i, :]) 


        T_eq, X_eq = comp_temp(T_guess_eq[i], X_guess, ps[8], eq_ratio, H_react,
                               H2, O2, N2, H2O, OH, O, H, NO)
        
        H_prod_f = rt.H_prod(T_eq, X_eq, eq_ratio, H2, O2, N2, H2O, OH, O, H, NO)

        X_matrix_eq[i, :] = X_eq
        T_final_eq[i] = T_eq

        print(f"---End results for Equivalence Ratio of {eq_ratio:.1f}")
        print(f"H_react = {H_react:.2f} J")
        print(f"H_prod({T_eq:.2f}) = {H_prod_f:.2f} J")
        print(f"Flame temperature = {T_eq:.2f} K")
        print(f"X_H2 = {X_eq[0]*100:.2f}%")
        print(f"X_O2 = {X_eq[1]*100:.2f}%")
        print(f"X_N2 = {X_eq[2]*100:.2f}%")
        print(f"X_H2O = {X_eq[3]*100:.2f}%")
        print(f"X_OH = {X_eq[4]*100:.2f}%")
        print(f"X_O = {X_eq[5]*100:.2f}%")
        print(f"X_H = {X_eq[6]*100:.2f}%")
        print(f"X_NO = {X_eq[7]*100:.2f}%")
        #exit()

    fig1, ax1 = plt.subplots()
    ax1.plot(eq_ratios, T_final_eq)
    ax1.set(xlabel="Equivalence Ratio $\Phi$", ylabel="$T_{ad}$ [K]",
           title="Effect of equivalence ratio on the Flame")
    ax1.grid()
    fig1.savefig('../figs/eq_ratio.png')
    
    for i, p in enumerate(ps):

        T_stoic = 2500

        H_react = rt.H_react(eq_ratios[3], H2, O2, N2) 
        
        X_guess = np.array(X_guess_eq[3, :]) 

        T_p, X_p = comp_temp(T_stoic, X_guess, p, eq_ratios[3], H_react,
                             H2, O2, N2, H2O, OH, O, H, NO)

        H_prod_f = rt.H_prod(T_p, X_p, eq_ratios[3], H2, O2, N2, H2O, OH, O, H, NO)

        X_matrix_p[i, :] = X_p
        T_final_p[i] = T_p

        print(f"---End results for Equivalence Ratio of {eq_ratio:.1f}")
        print(f"H_react = {H_react:.2f} J")
        print(f"H_prod({T_p:.2f}) = {H_prod_f:.2f} J")
        print(f"Flame temperature = {T_p:.2f} K")
        print(f"X_H2 = {X_p[0]*100:.2f}%")
        print(f"X_O2 = {X_p[1]*100:.2f}%")
        print(f"X_N2 = {X_p[2]*100:.2f}%")
        print(f"X_H2O = {X_p[3]*100:.2f}%")
        print(f"X_OH = {X_p[4]*100:.2f}%")
        print(f"X_O = {X_p[5]*100:.2f}%")
        print(f"X_H = {X_p[6]*100:.2f}%")
        print(f"X_NO = {X_p[7]*100:.2f}%")

    fig2, ax2 = plt.subplots()
    ax2.plot(ps, T_final_p)
    ax2.set(xlabel="Pressure [atm]", ylabel="$T_{ad}$ [K]",
           title="Effect of pressure on the Flame")
    ax2.grid()
    fig2.savefig('../figs/pressure.png')
    
    #plt.show()

    #for i in range(1):
    #    # Fixed point iterative method
    #    # From conservation of atoms
    #    # X_N2 = [3.76*(2*X_O2 + X_H2O + X_OH + X_O + X_H) - X_NO]/2
    #    X_k[2] = (1/2)*(3.76*(2*X_k_old[1] + X_k_old[3] + X_k_old[4] 
    #                    + X_k_old[5] + X_k_old[7]) - X_k_old[7])
    #    # X_H2 = [2*\Phi*(2*X_O2 + X_H2O + X_OH + X_O + X_H) 
    #    #        -2*X_H2O - X_OH - X_H]/2
    #    X_k[0] = (1/2)*(2*eq_ratio[3]*(2*X_k_old[1] + X_k_old[3] + X_k_old[4] 
    #                    + X_k_old[5] + X_k_old[7]) - 2*X_k_old[3] - X_k_old[4] 
    #                    - X_k_old[6])
    #    # X_O2 = 1 - \Sigma{X_i} # except for O2
    #    left_O2 = 1 - X_k[0] - X_k[2] - np.sum(X_k_old[3:])  
    #    if left_O2 < 0:
    #        X_k[1] = X_k_old[1] 
    #    else:
    #        X_k[1] = left_O2
    #    # From kp_F equations
    #    # X_H2O = kp_F_H2O*X_H2*\sqrt(X_O2)*\sqrt(p)
    #    X_k[3] = kp_F_H2O*X_k[0]*np.sqrt(X_k[1])*np.sqrt(p_ct)
    #    # X_OH = kp_F_OH*\sqrt{X_H2}*\sqrt{X_O2}
    #    X_k[4] = kp_F_OH*np.sqrt(X_k[0])*np.sqrt(X_k[1])
    #    # X_O = kp_F_O*\sqrt{X_O2}*\frac{1}{\sqrt{p}}
    #    X_k[5] = kp_F_O*np.sqrt(X_k[1])/np.sqrt(p_ct)
    #    # X_H = kp_F_H*\sqrt{X_H2}*\frac{1}{\sqrt{p}}
    #    X_k[6] = kp_F_H*np.sqrt(X_k[0])/np.sqrt(p_ct)
    #    # X_NO = kp_F_NO*\sqrt{X_H2}*\sqrt{X_O2}
    #    X_k[7] = kp_F_NO*np.sqrt(X_k[2])*np.sqrt(X_k[1])

    #    print('X', X_k)
    #    print(f"Sum of X = {np.sum(X_k)}")
    #    print(f"Error = {np.linalg.norm(X_k - X_k_old)}")
    #    X_k_old = X_k

