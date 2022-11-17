import numpy as np
import reactions as rt

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

def bisection(T_c, H_react, x_k, phi, H2, O2, N2, H2O, OH, O, H, NO, 
              error_accept, max_it, scaling):

    T_a = T_c - 500/scaling
    T_b = T_c + 500/scaling
    #print(f"---Bisection between {T_a} and {T_b}")

    it = 1
    error = abs(T_a-T_b)
    
    while (error > error_accept and it <= max_it):
        T_c = (T_a + T_b)/2
        #print(f"Iteration {it} > T_c = {T_c}")

        H_prod_T_a = rt.H_prod(T_a, x_k, phi, H2, O2, N2, H2O, OH, O, H, NO)  
        fa = H_react - H_prod_T_a 
        #print(f"f(T_a) = {fa}")
        
        H_prod_T_b = rt.H_prod(T_b, x_k, phi, H2, O2, N2, H2O, OH, O, H, NO)  
        fb = H_react - H_prod_T_b
        #print(f"f(T_b) = {fb}")

        H_prod_T_c = rt.H_prod(T_c, x_k, phi, H2, O2, N2, H2O, OH, O, H, NO)  
        fc = H_react - H_prod_T_c
        #print(f"f(T_c) = {fc}")

        if ((fa*fb) >= 0):
            print("Algorithm crash")
            it += 1
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
            it += 1

    H_prod_T_c = rt.H_prod(T_c, x_k, phi, H2, O2, N2, H2O, OH, O, H, NO)
    #print(f"---Converged H_prod({T_c}) = {H_prod_T_c} approximate to H_react = {H_react}")
    return T_c
