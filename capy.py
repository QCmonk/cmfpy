# -*- coding: utf-8 -*-
# @Author: Helios
# @Date:   2018-04-24 11:26:02
# @Last Modified by:   Spark
# @Last Modified time: 2018-07-11 14:35:27

# TODO 
# reexpress matched filter in terms of cvxpy


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
        self.svector = np.asarray(svector, dtype=np.float).reshape([1,-1])
        # dimensionality of measurement basis
        self.mdim = len(self.svector)
        # dimensionality of signal basis
        self.ndim = len(kwargs["transform"].T)
        # check for verbosity level
        self.verbose = verbose
        # sensing flags for debug purposes
        self.flags = []

        # extract keyword arguments after setting defaults
        self.opt_params = {"basis": 'random', "sparsity": 0.1,
                           "epsilon": 0.01, "freqs": [1e2, 1e3], "length": len(self.svector)}
        for key, value in kwargs.items():
            self.opt_params[key] = value


        # check for supplied measurement transform
        if "transform" not in self.opt_params.keys():
            raise KeyError("No transform specified, aborting")
        else:
            self.transform = self.opt_params["transform"]



    # method to perform optimisation
    def cvx_recon(self):
        # set cvx start flag
        self.flags.append("cvx_recon_start")

        # setup SDP using cvxpy
        if self.verbose: print("Setting up problem")
        A = self.transform 
        b = self.svector
        x = cvx.Variable(len(self.transform.T))
        objective = cvx.Minimize(cvx.norm(x, 1))
        constraints = [cvx.norm(A*x - b, 2) <= self.opt_params["epsilon"]]
        prob = cvx.Problem(objective, constraints)

        # solve 2nd order cone problem
        if self.verbose: print("Solving using CVXOPT")
        prob.solve()

        # print solution status
        if self.verbose:   
            print("Solution status:", prob.status)
            print("Objective: ", prob.value)


        # set cvx end flag
        self.flags.append("cvx_recon_end")
        # compute reconstruction
        self.u_recon = x.value
        # store error vector
        self.u_error = self.transform @ self.u_recon - self.svector
        # compute error using both l1 and l2 norm
        self.metrics = {'l1': np.linalg.norm(self.u_error, ord=1), "l2": np.linalg.norm(self.u_error, ord=2)}

################################################################################################################

    # find time index of most prominent spike using python optimisation
    def py_match(self, power=1, osignal=None, plot=False):
        # ensure a template has been provided
        if "template" not in self.opt_params.keys():
            raise(KeyError, "No template provided for matched filter")

        # set single shot matched filter flag
        self.flags.append("comp_match_single_start")
        # set time range
        tau_range = self.opt_params["time"]

        # status message
        if self.verbose: print("Performing compressive matched filtering of single spike")

        # define autocorrelation function
        tau_match = lambda tau_int: np.abs(self.svector @ self.transform @ self.sig_shift(self.opt_params["template"], tau_int))
        
        # preallocate correlation vector
        self.correlation = np.zeros((self.ndim,))
        for step, tau in enumerate(tau_range):
            # compute correlation given some time shift of template vector
            self.correlation[step] = tau_match(step)**power

        # set single shot matched filter flag
        self.flags.append("comp_match_single_end")

        # plot correlation if requested against original signal if supplied
        if plot:
            if osignal is not None:
                fig, axx = plt.subplots(2, sharex=True)
                axx[1].plot(tau_range, self.correlation, 'r')
                axx[0].grid(True)
                axx[0].set_title("Correlation using template")
                axx[1].set_xlabel("Time (s)")
                axx[0].plot(tau_range, osignal)
                axx[1].grid(True)
                plt.show()
            else:
                plt.plot(tau_range, self.correlation, 'r')
                plt.title("Correlation using template")
                plt.xlabel("Time (s)")
                plt.ylabel("Correlation")
                plt.grid(True)
                plt.show()


    # performs multi-event compressive sampling
    def py_notch_match(self, osignal=None, max_spikes=1, plot=False):
        # set multi match start
        self.flags.append('comp_match_multi_start')

        if self.verbose: print("Performing compressive matched filtering")

        # define notch function
        self.notch = np.ones((self.ndim,1), dtype=float)
        # time period to shift over
        tau_range = self.opt_params["time"]
        # define autocorrelation function
        tau_match = lambda tau_int: np.abs(self.svector @ self.transform @ np.multiply(self.notch ,self.sig_shift(self.opt_params["template"], tau_int)))
        
        spike = 0
        self.template_recon = np.zeros((self.ndim, 1))
        while spike < max_spikes:
            # iterate the number of spikes
            spike += 1

            # compute correlation function over parameter space
            self.correlation = np.zeros((self.ndim,), dtype=float)
            for step, tau in enumerate(tau_range):
                # test = self.transform @ np.multiply(self.notch, self.sig_shift(self.opt_params["template"], step))
                self.correlation[step] = tau_match(step)

            # extract maximum peak
            max_ind = np.argmax(a=self.correlation)
            # add peak to vector nuke
            self.notch[max_ind-len(self.opt_params["template"]):max_ind+2*len(self.opt_params["template"])] = 0
            # subtract template from measurement vector
            self.template_recon += self.sig_shift(self.opt_params["template"], max_ind).reshape([-1, 1])

        self.flags.append('comp_match_multi_end')

        # compute error
        self.m_error = self.svector-self.transform @ self.template_recon
        # compute errors
        self.metrics = {'l1': np.linalg.norm(self.m_error, ord=1), "l2": np.linalg.norm(self.m_error, ord=2)}

        # plot reconstruction if requested
        if plot:
            if osignal is not None:
                fig, axx = plt.subplots(2, sharex=True)
                axx[1].plot(tau_range, self.notch, 'r')
                axx[0].grid(True)
                axx[0].set_title("Multievent reconstruction")
                axx[1].set_xlabel("Time (s)")
                axx[0].plot(tau_range, osignal)
                axx[1].grid(True)
                plt.show()
            else:
                plt.plot(tau_range, self.template_recon, 'r')
                plt.title("Correlation using template")
                plt.xlabel("Time (s)")
                plt.ylabel("Correlation")
                plt.grid(True)
                plt.show()


    # computes an estimate for the amplitude given the best guess 
    def amp_get(self, max_ind, template):
        # should we use the full reconstruction or just the single spike?
        waveform = self._sig_shift(template, max_ind/self.opt_params["fs"]).reshape([-1, 1])
        # compute estimated amplitude
        t_waveform = self.transform.dot(waveform)
        print(np.shape(self.measurement))
        wave_amp = self.measurement.dot(t_waveform)/(np.linalg.norm(t_waveform ,2)**2)
        return wave_amp

    # shifts signal vector by tau
    def sig_shift(self, template, tau_int):
        # create zeroed vector (inefficent!)
        shift = np.zeros((self.ndim,1))
        temp_len = len(template)
        # force clipping
        if tau_int < 0:
            int_pos = 0
        elif tau_int > self.ndim - temp_len:
            tau_int = int(self.ndim - temp_len)
        # shift vector by specified amount
        shift[tau_int: tau_int + temp_len] = template.reshape([-1,1]) 

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
def measure_gen(ovector, time, basis="random", measurements=100):

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
