import os
import sys
import random
import numpy as np
from capy import *
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt


####################################################################
########################### Example usage ##########################
####################################################################

# generate an example sparse signal with len(events) signal events 
# set random seed for repeatability
np.random.seed(4321)
time, signal, template = sparse_gen(events=3, freq=1e12, fs=4e12, t=1e-9, plot=False)

# define a measurement basis (random or fourier)
transform = measure_gen(ovector=signal, time=time, basis="random")

# compute the signal that would be measured given the above original
# and measurement basis
svector = transform @ signal.T

# store relevant parameters of problem 
user_vars = {
		     # time samples of signal pre-transform
			 "time": time,
			 # template waveform to use for matched filter
			 "template": template,
			 # number of measurements used in transform basis
			 "measurements": len(transform),
			 # epsilon disc radius for noisy data
			 "epsilon": 0.01}

####################################################################

comp = CAOptimise(svector=svector, transform=transform, verbose=True, **user_vars)
recon = comp.cvx_recon()
#comp.py_notch_match(osignal=signal , plot=True, max_spikes=3)