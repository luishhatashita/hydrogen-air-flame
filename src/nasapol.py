import re

def find_element_lines(element):
    with open('../data/nasa7pol.dat', 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        value_lines = [lines[i] for i in range(5, len(lines)-1)]
        for line_num, line in enumerate(value_lines):
            if element == line.split()[0]:
                #print(value_lines[line_num:line_num+4])
                return value_lines[line_num:line_num+4]

def nasa_7_coeff(T, lines):
    # Preparing lines, according to the temperature line and coefficient lines
    T_split_line = lines[0].split()
    #print(T_split_line)
    coef_lines = [line for line in lines[1:]]
    coef_split_lines = [None]*3
    for i, coef_split_line in enumerate(coef_lines):
        coef_split_lines[i] = [coef_split_line[idx:idx+15] for idx in range(0, len(coef_split_line), 15)]
    
    # Defining temperature ranges
    size_T_split = len(T_split_line)
    T_low = float(T_split_line[size_T_split-4])
    T_high = float(T_split_line[size_T_split-3])
    T_mid = float(T_split_line[size_T_split-2])
    #if len(T_split_line) == 8:
    #    T_low = float(T_split_line[4])
    #    T_high = float(T_split_line[5])
    #    T_mid = float(T_split_line[6])
    #elif len(T_split_line) == 9:
    #    T_low = float(T_split_line[5])
    #    T_high = float(T_split_line[6])
    #    T_mid = float(T_split_line[7])
    #elif len(T_split_line) == 10:
    #    T_low = float(T_split_line[6])
    #    T_high = float(T_split_line[7])
    #    T_mid = float(T_split_line[8])

    # Assigning coefficients according to the correct temperature range
    if (T >= T_low and T <= T_mid):
        #print("lower interval")
        coefs_1 = [float(coef) for coef in coef_split_lines[1][2:5]] 
        coefs_2 = [float(coef) for coef in coef_split_lines[2][:4]] 
        coefs = coefs_1 + coefs_2
        #print(coefs)
        return coefs
    elif (T >= T_mid and T <= T_high):
        #print("upper interval")
        coefs_1 = [float(coef) for coef in coef_split_lines[0][:5]] 
        coefs_2 = [float(coef) for coef in coef_split_lines[1][:2]] 
        coefs = coefs_1 + coefs_2
        #print(coefs)
        return coefs
    else:
        print("Out of temperature range")

#if __name__ == '__main__':
#    T = 1500 
#    element = 'H'
#    element_lines = find_element_lines(element)
#    coefficients = nasa_7_coeff(T, element_lines)
