import nasapol as nasa
import numpy as np

class gas():
    def __init__(self, specie):
        self.name = specie
        self.lines = nasa.find_element_lines(specie)
        self.T_low, self.T_mid, self.T_high, self.coefs_low, self.coefs_high = nasa.nasa_7_coeff(self.lines)

    def print_coefs(self):
        print("Lower range coefs\n", self.coefs_low)
        print("Higher range coefs\n", self.coefs_high)

    def get_enthalpy_RT(self, T):
        T_array = np.array([1, T/2, (T**2)/3, (T**3)/4, (T**4)/5, 1/T]) 

        if (T >= self.T_low and T<= self.T_mid):
            h_RT = np.dot(self.coefs_low[:-1], T_array)
        elif (T >= self.T_mid and T<= self.T_high):
            h_RT = np.dot(self.coefs_high[:-1], T_array)
        else:
            print("Out of temperature range.")
            pass

        return h_RT
    
    def get_entropy_R(self, T):
        T_array = np.array([np.log(T), T, (T**2)/2, (T**3)/3, (T**4)/4, 1]) 

        if (T >= self.T_low and T<= self.T_mid):
            tmp_coefs = np.append(self.coefs_low[:-2], self.coefs_low[-1])
            s_R = np.dot(tmp_coefs, T_array)
        elif (T >= self.T_mid and T<= self.T_high):
            tmp_coefs = np.append(self.coefs_high[:-2], self.coefs_high[-1])
            s_R = np.dot(tmp_coefs, T_array)
        else:
            print("Out of temperature range.")
            pass

        return s_R
    
    def get_gibbs_RT(self, T):
        return (self.get_enthalpy_RT(T) - self.get_entropy_R(T))
