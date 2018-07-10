import os
import sys
import random
import numpy as np
import matlab.engine
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt



# An example waveform that is sparse in time over a one second period
def sparse_gen(events, freq, fs=2e5, t=1.0, plot=False):
	

	# define rectangular function centered at 0 with width equal to period
	def rect(period, time):
		return np.where(np.abs(time) <= period/2, 1, 0)

	# initialise time vector
	time = np.arange(0,t,1/fs)

	# generate signal with a single sinusoid of frequency freq
	signal = np.zeros((len(time)))
	for evnt in events:
		signal += np.multiply(rect(1/freq, time-evnt),np.sin(2*np.pi*freq*(time-evnt)))

	# plot the source signal if requested
	if plot:
		plt.plot(time, signal, 'r--')
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.title("Initial sparse signal")
		plt.grid(True)
		plt.show()

	return time, signal

sparse_gen([1e-1,3e-1,6e-1], freq=1000, plot=True)

#comp = CAOptimise(svector=signal(time).reshape([-1,1]), verbose=True, **user_vars)
