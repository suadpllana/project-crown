import numpy as np

# Decay constants (Jaffey et al. 1971), per year, and present-day 238U/235U (Hiess et al. 2012).
LAMBDA_238 = 1.55125e-10
LAMBDA_235 = 9.8485e-10
U238_U235 = 137.818
YEARS_PER_MA = 1.0e6
# Age window (Ma) in which concordia intercepts are sought.
T_MIN_MA = -1000.0
T_MAX_MA = 5000.0
