import re

def find_element_lines(element):
    with open('../data/nasa7pol.dat', 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        value_lines = [lines[i] for i in range(5, len(lines)-1)]
        for line_num, line in enumerate(value_lines):
            if element == line.split()[0]:
                #print(value_lines[line_num:line_num+4])
                return value_lines[line_num:line_num+4]

def nasa_7_coeff(lines):
    # Preparing lines, according to the temperature line and coefficient lines
    T_split_line = lines[0].split()
    #print(T_split_line)
    coef_lines = [line for line in lines[1:]]
    coef_split_lines = [None]*3
    for i, coef_split_line in enumerate(coef_lines):
        coef_split_lines[i] = [coef_split_line[idx:idx+15] 
                               for idx in range(0, len(coef_split_line), 15)]
    
    # Defining temperature ranges
    size_T_split = len(T_split_line)
    T_low = float(T_split_line[size_T_split-4])
    T_high = float(T_split_line[size_T_split-3])
    T_mid = float(T_split_line[size_T_split-2])

    # Assigning coefficients according to the correct temperature range
    #print("lower interval")
    coefs_1_low = [float(coef) for coef in coef_split_lines[1][2:5]] 
    coefs_2_low = [float(coef) for coef in coef_split_lines[2][:4]] 
    coefs_low = coefs_1_low + coefs_2_low
    #print("upper interval")
    coefs_1_high = [float(coef) for coef in coef_split_lines[0][:5]] 
    coefs_2_high = [float(coef) for coef in coef_split_lines[1][:2]] 
    coefs_high = coefs_1_high + coefs_2_high
    return T_low, T_mid, T_high, coefs_low, coefs_high
