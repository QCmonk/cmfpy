import os
import sys
import random
import numpy as np
from capy import *
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt



# An example waveform that is sparse in time over a one second period
def sparse_gen(events, freq, fs=4e3, t=10, plot=False):
	

	# define rectangular function centered at 0 with width equal to period
	def rect(period, time):
		return np.where(np.abs(time) <= period/2, 1, 0)

	# initialise time vector
	time = np.arange(0,t,1/fs)

	# generate signal with a single sinusoid of frequency freq
	signal = np.zeros((len(time)))
	for evnt in np.random.choice(time, size=events):
		signal += np.multiply(rect(1/freq, time-evnt),-np.sin(2*np.pi*freq*(time-evnt)))

	# generate template signal for match
	t = np.arange(0, 1/freq, 1/fs)
	template = np.sin(2*np.pi*freq*t)

	# plot the source signal if requested
	if plot:
		plt.plot(time, signal, 'r--')
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.title("Initial sparse signal")
		plt.grid(True)
		plt.show()

	return time, signal, template

####################################################################
########################### Example usage ##########################
####################################################################

# generate an example sparse signal with len(events) signal events 
# set random seed for repeatability
np.random.seed(4321)
time, signal, template = sparse_gen(events=3, freq=1e12, fs=4e12, t=1e-6, plot=False)

# define a measurement basis (random or fourier)
transform = measure_gen(ovector=signal, time=time, basis="random")

# compute the signal that would be measured given the above original
# and measurement basis
svector = transform @ signal.T

# store relevant parameters of problem 
user_vars = {"transform": transform, 
		     # time samples of signal pre-transform
			 "time": time,
			 # template waveform to use for matched filter
			 "template": template,
			 # number of measurements used in transform basis
			 "measurements": len(transform),
			 # epsilon disc radius for noisy data
			 "epsilon": 0.01}
####################################################################

comp = CAOptimise(svector=svector, verbose=True, **user_vars)
comp.py_notch_match(osignal=signal , plot=True, max_spikes=3)