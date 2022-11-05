import nasapol as nasa
import reactions as rt 
import gas as phase 
import cantera as ct
import numpy as np

def F(x, p, phi, kp_vector):
    # Let X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    # in variable form: X = (x1, x2, x3, x4, x5, x6, x7, x8)^T

    # For the following set of equations:
    # f1 := (p^(-1/2))*x4/(x1*x2^(1/2)) - kp_F_H2O(T) = 0
    # f2 := x5/(x1^(1/2)*x2^(1/2)) - kp_F_OH(T) = 0
    # f3 := (p^(1/2))*x6/(x2^(1/2)) - kp_F_O(T) = 0
    # f4 := (p^(1/2))*x7/(x1^(1/2)) - kp_F_H(T) = 0
    # f5 := x8/(x2^(1/2)*x3^(1/2)) - kp_F_NO(T) = 0
    # f6 := \frac{2x1+2x4+x5+x7}{2x2+x4+x5+x6+x8} - 2\Phi = 0 
    # f7 := \frac{2x3+x8}{2x2+x4+x5+x6+x8} - 3.76 = 0 
    # f8 := \Sigma_{i=1}^{8}xi - 1 = 0 

    # Additionally, the kp_vector is defined as:
    # kp_vector = [kp_F_H2O(T), kp_F_OH(T), kp_F_O(T), kp_F_H(T), 
    #              kp_F_NO(T)]

    F = np.zeros([8,1]) # column vector 8x1

    # First five equations are from the formation reactions of the species not
    # in their natural form, e.g., OH, O, H, etc.
    F[0, 0] = (p**(-1/2))*x[3]/(x[0]*(x[1]**(1/2))) - kp_vector[0]
    F[1, 0] = x[4]/((x[0]**(1/2))*(x[1]**(1/2))) - kp_vector[1]
    F[2, 0] = (p**(1/2))*x[5]/(x[1]**(1/2)) - kp_vector[2]
    F[3, 0] = (p**(1/2))*x[6]/(x[0]**(1/2)) - kp_vector[3]
    F[4, 0] = x[7]/((x[1]**(1/2))*(x[3]**(1/2))) - kp_vector[4]

    # Ratio of numbers of atoms from conservation to express in terms of molar
    # fractions.
    F[5, 0] = (2*x[0]+2*x[3]+x[4]+x[6])/(2*x[1]+x[3]+x[4]+x[5]+x[7]) - 2*phi
    F[6, 0] = (2*x[2]+x[7])/(2*x[1]+x[3]+x[4]+x[5]+x[7]) - 3.76

    # From molar fraction definition.
    F[7, 0] = np.sum(x) - 1 

    return F

def jacobian(x, p):
    # Let X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    # in variable form: X = (x1, x2, x3, x4, x5, x6, x7, x8)^T

    # For the following set of equations:
    # f1 := (p^(-1/2))*x4/(x1*x2^(1/2)) - kp_F_H2O(T) = 0
    # f2 := x5/(x1^(1/2)*x2^(1/2)) - kp_F_OH(T) = 0
    # f3 := (p^(1/2))*x6/(x2^(1/2)) - kp_F_O(T) = 0
    # f4 := (p^(1/2))*x7/(x1^(1/2)) - kp_F_H(T) = 0
    # f5 := x8/(x2^(1/2)*x3^(1/2)) - kp_F_NO(T) = 0
    # f6 := \frac{2x1+2x4+x5+x7}{2x2+x4+x5+x6+x8} - 2\Phi = 0 
    # f7 := \frac{2x3+x8}{2x2+x4+x5+x6+x8} - 3.76 = 0 
    # f8 := \Sigma_{i=1}^{8}xi - 1 = 0 

    J = np.zeros([8,8])
    #J = np.zeros(8,8, dtype=np.longdouble)
    
    # Defining first line as:
    # J[0, i] = \frac{f1}{xi}, with the following non-zero terms:
    J[0, 0] = -(p**(-1/2))*x[3]/((x[0]**2)*(x[1]**(1/2)))
    J[0, 1] = -(p**(-1/2))*x[3]/(2*x[0]*(x[1]**(3/2)))
    J[0, 3] = (p**(-1/2))/(x[0]*(x[1]**(1/2)))

    # Defining second line as:
    # J[1, i] = \frac{f2}{xi}, with the following non-zero terms:
    J[1, 0] = -x[4]/(2*(x[0]**(3/2))*(x[1]**(1/2)))
    J[1, 1] = -x[4]/(2*(x[0]**(1/2))*(x[1]**(3/2)))
    J[1, 4] = 1/((x[0]**(1/2))*(x[1]**(1/2)))

    # Defining third line as:
    # J[2, i] = \frac{f3}{xi}, with the following non-zero terms:
    J[2, 1] = -(p**(1/2))*x[5]/(2*(x[1]**(3/2)))
    J[2, 5] = (p**(1/2))/((x[1]**(1/2)))

    # Defining forth line as:
    # J[3, i] = \frac{f4}{xi}, with the following non-zero terms:
    J[3, 0] = -(p**(1/2))*x[6]/(2*(x[1]**(3/2)))
    J[3, 6] = (p**(1/2))/((x[0]**(1/2)))

    # Defining fifth line as:
    # J[4, i] = \frac{f5}{xi}, with the following non-zero terms:
    J[4, 1] = -x[7]/(2*(x[1]**(3/2))*(x[2]**(1/2)))
    J[4, 2] = -x[7]/(2*(x[1]**(1/2))*(x[2]**(3/2)))
    J[4, 7] = 1/((x[1]**(1/2))*(x[2]**(1/2)))

    # Defining sixth line as:
    # J[5, i] = \frac{f5}{xi}, with the following non-zero terms:
    J[5, 0] = 2/(2*x[1]+x[3]+x[4]+x[5]+x[7]) 
    J[5, 1] = -(2*(2*x[0]+2*x[3]+x[4]+x[6]))/((2*x[1]+x[3]+x[4]+x[5]+x[7])**2) 
    J[5, 3] = ((-2*x[0]+4*x[1]+x[4]+2*x[5]-x[6]+2*x[7])
               /((2*x[1]+x[3]+x[4]+x[5]+x[7])**2)) 
    J[5, 3] = ((-2*x[0]+2*x[1]-x[3]+x[5]-x[6]+x[7])
               /((2*x[1]+x[3]+x[4]+x[5]+x[7])**2)) 
    J[5, 5] = J[5, 1]/2 
    J[5, 6] = J[5, 0]/2 
    J[5, 7] = J[5, 1]/2 

    # Defining seventh line as:
    # J[6, i] = \frac{f7}{xi}, with the following non-zero terms:
    J[6, 1] = -(2*(2*x[2]+x[7]))/((2*x[1]+x[3]+x[4]+x[5]+x[7])**2) 
    J[6, 2] = 2/((2*x[1]+x[3]+x[4]+x[5]+x[7])) 
    J[6, 3] = J[6, 1]/2
    J[6, 4] = J[6, 1]/2
    J[6, 5] = J[6, 1]/2
    J[6, 7] = (2*x[1]-2*x[2]+x[3]+x[4]+x[5])/((2*x[1]+x[3]+x[4]+x[5]+x[7])**2) 

    # Defining eigth line as:
    # J[7, i] = \frac{f8}{xi} = 1
    J[7, :] = 1

    return J

def H_prod_RT(T, X, H2, O2, N2, H2O, OH, O, H, NO):
    # Let X = [X_H2, X_O2, X_N2, X_H2O, X_OH, X_O, X_H, X_NO]
    H_RT_vect = [H2.get_enthalpy_RT(T), O2.get_enthalpy_RT(T), 
                 N2.get_enthalpy_RT(T), H2O.get_enthalpy_RT(T),
                 OH.get_enthalpy_RT(T), O.get_enthalpy_RT(T), 
                 H.get_enthalpy_RT(T), NO.get_enthalpy_RT(T)]

    #print("H_prod_RT: ", np.dot(X, H_RT_vect))
    return np.dot(X, H_RT_vect)

def bisection(T_c, H_react_RT, x_k, H2, O2, N2, H2O, OH, O, H, NO, 
              error_accept, max_it):

    T_a = T_c - 200
    T_b = T_c + 200
    print(f"Bisection between {T_a} and {T_b}")

    it = 1
    error = abs(T_a-T_b)
    
    while (error > error_accept and it <= max_it):
        T_c = (T_a + T_b)/2
        print(T_c)

        H_prod_RT_T_a = H_prod_RT(T_a, x_k, H2, O2, N2, H2O, OH, O, H, NO)  
        fa = H_react_RT - H_prod_RT_T_a 
        print(f"f(T_a) = {fa}")
        
        H_prod_RT_T_b = H_prod_RT(T_b, x_k, H2, O2, N2, H2O, OH, O, H, NO)  
        fb = H_react_RT - H_prod_RT_T_b
        print(f"f(T_b) = {fb}")

        H_prod_RT_T_c = H_prod_RT(T_c, x_k, H2, O2, N2, H2O, OH, O, H, NO)  
        fc = H_react_RT - H_prod_RT_T_c
        print(f"f(T_c) = {fc}")

        if ((fa*fb) >= 0):
            print("Algorithm crash")
        elif ((fc*fa) < 0):
            T_b = T_c
            error = abs(T_a-T_b)
            it += 1
        elif ((fc*fb) < 0):
            T_a = T_c
            error = abs(T_a-T_b)
            it += 1
        else:
            print("Unknown error")

        return T_c

if __name__ == '__main__':

    # State properties
    p_ct = 20 #* ct.one_atm
    
    # Define equivalence ration interval 
    N = int((1.3-0.7)/0.1 + 1)
    eq_ratio = np.linspace(0.7, 1.3, num=N)
    
    # Iterative method tolerance
    max_it_b = 10
    error_norm = 0.01
    error_T = 10

    species = ['H2', 'O2', 'N2', 'H2O', 'OH', 'O', 'H', 'NO']
    print(species)
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
    X_k_old = np.array([0.01, 30/(1e6), 0.59, 0.30, 1300/(1e6), 30/(1e6), 2600/(1e6), 3/(1e9)])
    #X_k_old = np.array([0.3, 0.2, 0.5, 0, 0, 0, 0, 0])
    #X_k_old = (1/8)*np.ones(8) 
    #print(X_k_old.shape)
    print(f"Sum X_k_old: {np.sum(X_k_old)}")
    T_k_old = 2500
    #print(jacobian(X_k_old, p_ct))
    X_k = np.zeros(8)

    for i in range(5):
        
        # Reactants calculation of initial flame composition
        n_tot = eq_ratio[3] + 0.5 + (3.76/2)
        H_react_RT = ((eq_ratio[3]/n_tot)*H2.get_enthalpy_RT(298) 
                      + (0.5/n_tot)*O2.get_enthalpy_RT(500) 
                      + ((3.76/2)/n_tot)*N2.get_enthalpy_RT(500))
        print(f"H_react_RT: {H_react_RT}")

        # Reaction values:
        kp_F_H2O = rt.kp(rt.dG_RT_F_H2O(T_k_old, H2, O2, H2O))
        kp_F_OH = rt.kp(rt.dG_RT_F_OH(T_k_old, H2, O2, OH))
        kp_F_O = rt.kp(rt.dG_RT_F_O(T_k_old, O2, O))
        kp_F_H = rt.kp(rt.dG_RT_F_H(T_k_old, H2, H))
        kp_F_NO = rt.kp(rt.dG_RT_F_NO(T_k_old, N2, O2, NO))

        kp_vector = np.array([kp_F_H2O, kp_F_OH, kp_F_O, kp_F_H, kp_F_NO])

        # Euler's method for:
        # x_k = x_k-1 - J^-1(x_k-1)F(x_k-1), 
        # given J as the jacobian of the F function.
        j_inv = np.linalg.inv(jacobian(X_k_old, p_ct))
        F_X_k_old = F(X_k_old, p_ct, eq_ratio[3], kp_vector)
        j_inv_F_X_k_old = np.matmul(j_inv, F_X_k_old).T
        j_inv_F_X_k_old = np.squeeze(j_inv_F_X_k_old)
        #print("J^-1F = ", j_inv_F_X_k_old)
        #print(j_inv_F_X_k_old.shape)
        X_k = np.add(X_k_old, -j_inv_F_X_k_old)

        # Verifying the validity of the results
        print('X = ', X_k)
        print(f"Sum of X = {np.sum(X_k)}")
        print(f"Error = {np.linalg.norm(X_k - X_k_old)}")

        T_k_old = bisection(T_k_old, H_react_RT, X_k, H2, O2, N2, 
                            H2O, OH, O, H, NO, error_T, max_it_b)
        X_k_old = X_k

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

