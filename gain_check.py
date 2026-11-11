# gain_check.py — put beside dpp.py
import numpy as np
from dpp import trapezoidal_filter, pz_parameter, filter_gain
fs, tau, k, l = 50e6, 50e-6, 200, 250
M = pz_parameter(tau, fs)
v = np.exp(-np.arange(5000) / (fs * tau))     # unit preamp step
h = trapezoidal_filter(v, k, l, M)[k + 25]
print("flat top / filter_gain =", h / filter_gain(k, M))   # want 1.0000

""" Author: Peleyeju Ayodele Humphrey
M.Sc. Physics (Electronics & Communications), Federal University Otuoke 13 Oct 2026. 

"""