# -*- coding: utf-8 -*-
# @Author: Helios
# @Date:   2018-04-24 11:26:02
# @Last Modified by:   Spark
# @Last Modified time: 2018-07-10 18:20:35

# TODO 
# replace matlab CVX with CVXOPT
# remove all atomic physics stuff


import os
import sys
import random
import numpy as np
import cvxpy as cvx
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt



class CAOptimise(object):
    # class to handle compressive measuring with a suite of possible
    # experiment parameters

    def __init__(self, svector, verbose=False, **kwargs):
        # the measured vector obtained from experiment
        self.svector = np.asarray(svector, dtype=np.float)
        # dimensionality of measurement basis
        self.mdim = len(self.svector)
        # check for verbosity level
        self.verbose = verbose
        # sensing flags
        self.flags = []

        # extract keyword arguments after setting defaults
        self.opt_params = {"basis": 'random', "sparsity": 0.1,
                           "epsilon": 0.01, "freqs": [1e2, 1e3], "length": len(self.svector)}
        for key, value in kwargs.items():
            self.opt_params[key] = value


        # # compute minimum number of measurements required to converge on
        # # perfect reconstruction
        # self.min_meas = np.ceil(2*self.opt_params["sparseness"]*np.log(
        #     self.mdim/self.opt_params["sparseness"]) + 5*self.opt_params["sparseness"]/4, )

        # # default number of measurements to 10% over minimum number of
        # if "measurements" not in self.opt_params:
        #     self.opt_params["measurements"] = int(self.min_meas*1.1)

        # # verbosity
        # if self.verbose:
        #     print("Minimum number of measurements for reconstruction: {}, Specified: {}".format(
        #         self.min_meas, self.opt_params["measurements"]))

        # check for supplied measurement transform
        if "transform" not in self.opt_params.keys():
            raise KeyError("No transform specified, aborting")
        else:
            self.transform = self.opt_params["transform"]

        self.cvx_recon()


    # method to perform optimisation
    def cvx_recon(self):
        # set cvx flag
        self.flags.append("cvx_recon")

        

        # setup SDP using cvxpy
        if self.verbose: print("Setting up convex optimisation problem")
        A = self.transform 
        b = self.svector
        x = cvx.Variable(len(self.transform.T))
        objective = cvx.Minimize(cvx.norm(x, 1))
        constraints = [cvx.norm(A*x - b, 2) <= self.opt_params["epsilon"]]
        prob = cvx.Problem(objective, constraints)

        # solve 2nd order cone problem
        if self.verbose: print("Solving LP using CVXOPT")
        prob.solve()

        exit()

        # compute reconstruction
        self.u_recon = np.asarray(self.engine.atomic_cvx(measurement, matlab_transform, self.opt_params["epsilon"]))

        # store error vector
        self.u_error = self.svector-self.u_recon
        # compute error using both l1 and l2 norm
        self.metrics = {'l1': np.linalg.norm(self.u_error, ord=1), "l2": np.linalg.norm(self.u_error, ord=2)}

    # find time index of most prominent spike using python optimisation
    def py_match(self, time, power=1):
        # set single shot matched filter flag
        self.flags.append("matched")

        # set time range
        tau_range = np.arange(0, self.opt_params["time"], 1/self.opt_params["fs"])

        if self.verbose: print("Performing compressive matched filtering")

        # generate noisy measured vector and convert to matlab type double
        if self.opt_params["noise"]:
            self.measurement = (np.dot(self.transform, self.svector) + self.opt_params["noise"]*np.random.rand(self.opt_params["measurements"], 1)).T
        else:
            self.opt_params["epsilon"] = 0.0
            self.measurement = np.dot(self.transform, self.svector).T

        # define autocorrelation function
        tau_match = lambda tau: np.abs(self.measurement.dot(self.transform.dot(self._sig_shift(self.opt_params["template"], tau))))
        self.correlation = np.zeros((self.mdim,))
        for step, tau in enumerate(tau_range):
            self.correlation[step] = tau_match(tau)**power

        # plot correlation
        fig, axx = plt.subplots(2, sharex=True)
        axx[1].plot(tau_range, self.correlation/np.max(self.correlation), 'r')
        axx[0].grid(True)
        axx[0].set_title("Correlation using neural template")
        axx[1].set_xlabel("Time (s)")
        axx[0].plot(time, self.svector)
        axx[1].grid(True)
        plt.show()


    # computes the position of 
    def py_notch_match(self, power=1, max_spikes=1):
        self.flags.append('atomic')

        if self.verbose: print("Performing compressive matched filtering")

        # define nuke function
        self.nuke = np.ones((self.mdim,1), dtype=float)
        # define autocorrelation function
        tau_match = lambda template, tau: np.abs(np.real(self.measurement.dot(self.transform.dot(np.multiply(self.nuke, self._sig_shift(template, tau))))))
        
        spike = 0
        self.template_recon = np.zeros((self.mdim, 1))
        while spike < max_spikes:
            spike += 1

            # compute correlation function over parameter space
            self.correlation = np.zeros((self.mdim,), dtype=np.complex64)
            for step, tau in enumerate(np.arange(0, self.opt_params["time"], 1/self.opt_params["fs"])):
                test = self.transform.dot(np.multiply(self.nuke, self._sig_shift(self.opt_params["template"], tau)))
                self.correlation[step] = tau_match(self.opt_params["template"], tau)**power

            # extract maximum peak
            max_ind = np.argmax(a=self.correlation)
            # add peak to vector nuke
            self.nuke[max_ind-len(self.opt_params["template"]):max_ind+2*len(self.opt_params["template"])] = 0
            # subtract template from measurement vector
            #self.measurement -= self.transform.dot(self._sig_shift(self.opt_params["template"], max_ind/self.opt_params["fs"])).T
            self.template_recon += self._sig_shift(self.opt_params["template"], max_ind/self.opt_params["fs"]).reshape([-1, 1])

        # compute error
        self.m_error = self.svector-self.template_recon
        # compute errors
        self.metrics = {'l1': np.linalg.norm(self.m_error, ord=1), "l2": np.linalg.norm(self.m_error, ord=2)}


    # comptues an estimate for the amplitude given the best guess 
    def amp_get(self, max_ind, template):
        # should we use the full reconstruction or just the single spike?
        waveform = self._sig_shift(template, max_ind/self.opt_params["fs"]).reshape([-1, 1])
        # compute estimated amplitude
        t_waveform = self.transform.dot(waveform)
        print(np.shape(self.measurement))
        wave_amp = self.measurement.dot(t_waveform)/(np.linalg.norm(t_waveform ,2)**2)
        return wave_amp

    # shifts signal vector by tau
    def _sig_shift(self, template, tau):
        # create zeroed vector (inefficent!)
        shift = np.zeros((self.mdim,1))
        int_pos = int(np.round(tau*self.opt_params["fs"]))
        temp_len = len(template)
        # force clipping
        if int_pos < 0:
            int_pos = 0
        elif int_pos > self.mdim - temp_len:
            int_pos = int(self.mdim - temp_len)
        # shift vector by specified amount
        shift[int_pos: int_pos + temp_len] = template.reshape([-1,1]) 

        return shift

    # basic plot code for the reconstructed signal
    def plot_recon(self, time):

        print(np.shape(time), np.shape(self.u_recon))
        plt.plot(time, self.svector, color="r", label='Original')
        plt.plot(time, self.u_recon, 'g', label='Reconstruction')
        plt.title("Samples: {}, Measurements: {}, Basis: {}, Noise Amplitude: {}, ||Ax - b||_2 = {:2e}".format(self.opt_params["length"],
                                                                                       self.opt_params["measurements"],
                                                                                       self.opt_params["basis"],
                                                                                       self.opt_params["noise"],
                                                                                       self.metrics["l2"]))
        plt.grid(True)
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.show()

    # plot the results of the notched matched filter
    def notch_match_plot(self, time):
        fig, axx = plt.subplots(2, sharex=True)
        l1, = axx[0].plot(time, self.template_recon, 'g')
        axx[0].grid(True)
        axx[0].set_title("Notched matched filter - Samples: {}, Measurements: {}, Basis: {}, Noise Amplitude: {}, :L_2 Error: {}".format(self.opt_params["length"],
                                                                                       self.opt_params["measurements"],
                                                                                       self.opt_params["basis"],
                                                                                       self.opt_params["noise"], 
                                                                                       self.metrics["l2"]))                                     
        l2, = axx[0].plot(time, self.svector, 'r')
        #plt.legend([l1,l2], ["Reconstructed", "Original"])
        l3, = axx[1].plot(time, self.nuke, 'b')
        axx[1].grid(True)
        axx[1].set_xlabel("Time (s)")
        plt.show()


# generates desired measurement basis set with given parameters
def measure_gen(ovector, time, basis="random", measurements=50):

    # store signal vector dimension
    mdim = len(ovector)

    if basis == "random":
        transform = 2*np.random.ranf(size=(measurements, mdim))-1

    elif basis == "fourier":
        # pre-allocate transform matrix
        transform = np.zeros((measurements, mdim), dtype=float)

        # predefine measurement steps (we always assume a one second
        # measurement period)

        # generate random frequencies or use those in freq list
        rand_flag = len(self.opt_params["freqs"]) < self.opt_params["measurements"]
        # create storage container for selected indices
        if not rand_flag:
            rand_ints = []

        meas_freq = []
        for i in range(self.opt_params["measurements"]):
            if rand_flag:
                # choose a random frequency over the given range
                freq = (self.opt_params["freqs"][1] - self.opt_params["freqs"][0])*np.random.ranf()+self.opt_params["freqs"][0]
            else:
                # choose a random frequency in the provided set and save index of chosen int
                randint = np.random.randint(low=0, high=len(self.opt_params["freqs"]))
                # ensure frequency has not been chosen before (inefficient but in the scheme of things, unimportant)
                while randint in self.rand_ints:
                    randint = np.random.randint(low=0, high=len(self.opt_params["freqs"]))
                # save chosen frequency
                self.rand_ints.append(randint)
                # add to set
                freq = self.opt_params["freqs"][randint]
            # add frequency to selection
            self.opt_params["meas_freq"].append(freq)
            transform[i, :] = -np.sin(2*np.pi*freq*self.t) #np.imag(np.exp(-1j*2*np.pi*freq*self.t))

    else:
        print("unknown measurement basis specified: exiting ")
        os._exit(1)
    return transform

def pulse_gen(freq=1/2e-3, tau=[3.0], amp=1):
    """
    generates a multipulse magnetic field signal
    """

    # define box function
    def box(t, start, end):
        if type(t) is not list or type(t) is not np.array:
            t = np.asarray(t)
        return (t > start) & (t < end)

    # define sum function method
    def sig_sum(t, terms):
        return sum(f(t) for f in terms)

    # list to hold spike function
    terms = []

    for time in tau:
        # generate sin template vector
        terms.append(lambda t, tau=time: box(t, tau, tau + 1/freq)*amp*np.sin(2*np.pi*freq*(t-tau)))
   
    signal = lambda t, funcs=terms: sig_sum(t, funcs)

    # return generated signal function, template used and time vector
    return signal
