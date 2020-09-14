# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:56:07 2017
Standart Atmosphere Calculator
@author: Dan
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

T_isa = np.array([288.15,  216.649, 216.649, 228.649, 270.65, 270.65,   214.649,  186.6499,  186.6499, 186.6499])
h_isa = np.array([0,       11000,   20000,   32000,   47000,  51000,    71000,    85000,     1000000, np.inf])
P_isa = np.array([100000,  22625,   5471.93, 867.254, 110.76, 66.84829, 3.949,    0.362,     0., 0.])

T_interp = interpolate.interp1d(h_isa,T_isa, fill_value = "extrapolate")
P_interp = interpolate.interp1d(h_isa,P_isa, fill_value = "extrapolate")

g = 9.80665
R = 287.00
def cal(p0, t0, a, h0, h1):
	if a != 0:
		t1 = t0 + a * (h1 - h0)
		p1 = p0 * (t1 / t0) ** (-g / a / R)
	else:
		t1 = t0
		p1 = p0 * math.exp(-g / R / t0 * (h1 - h0))
	return t1, p1

def density_isa(altitude):
	return P_interp(altitude)/(R*T_interp(altitude))
