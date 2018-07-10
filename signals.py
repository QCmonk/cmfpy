import os
import sys
import random
import numpy as np
from capy import *
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt



# An example waveform that is sparse in time over a one second period
def sparse_gen(events, freq, fs=4e3, t=0.5, plot=False):
	

	# define rectangular function centered at 0 with width equal to period
	def rect(period, time):
		return np.where(np.abs(time) <= period/2, 1, 0)

	# initialise time vector
	time = np.arange(0,t,1/fs)

	# generate signal with a single sinusoid of frequency freq
	signal = np.zeros((len(time)))
	for evnt in events:
		signal += np.multiply(rect(1/freq, time-evnt),-np.sin(2*np.pi*freq*(time-evnt)))

	# plot the source signal if requested
	if plot:
		plt.plot(time, signal, 'r--')
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.title("Initial sparse signal")
		plt.grid(True)
		plt.show()

	return time, signal

####################################################################
########################### Example usage ##########################
####################################################################

# generate an example sparse signal with len(events) signal events 
time, signal = sparse_gen(events=[2.5e-1], freq=1000, plot=False)

# define a measurement basis (random or fourier)
transform = measure_gen(ovector=signal, time=time)

# compute the signal that would be measured given the above original
# and measurement basis
svector = transform @ signal.T

# store relevant parameters of problem 
user_vars = {"transform": transform,
			 "measurements": len(transform),
			 "basis": "random",
			 "epsilon": 0.01}
####################################################################

comp = CAOptimise(svector=svector, verbose=True, **user_vars)
