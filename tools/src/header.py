import numpy as np

# Physical constants (CODATA 2018, exact or recommended values), SI units.
H_PLANCK = 6.62607015e-34        # J s
C_LIGHT = 299792458.0            # m s^-1
K_BOLTZMANN = 1.380649e-23       # J K^-1
AMU = 1.66053906660e-27          # kg
HARTREE = 4.3597447222071e-18    # J
BOHR = 5.29177210903e-11         # m

# Second radiation constant h c / k_B expressed in cm K (wavenumbers in cm^-1).
C2_CM_K = H_PLANCK * C_LIGHT * 100.0 / K_BOLTZMANN
