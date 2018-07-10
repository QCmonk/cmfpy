# -*- coding: utf-8 -*-
# @Author: Helios
# @Date:   2018-04-24 11:26:02
# @Last Modified by:   joshm
# @Last Modified time: 2018-07-05 17:02:27


import os
import sys
import random
import numpy as np
import scipy.sparse as sp
from decimal import Decimal
import matplotlib.pyplot as plt


# TODO 
# replace CVX with pyconv 
# improve the measurement array parsing and generation - too opaque right now   

class CAOptimise(object):
    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    engine : matlab.engine
        An instance of the matlab engine. Soon to be replaced.

    verbose : bool
        Whether to print progress reports as reconstruction is performed.

    kwargs : {"basis": "random", "sparsity": 0.1, "epsilon": 0.01
              "freqs": [1e2, 1e3], "length": len(s vector)}, optional
        optional arguments for reconstruction. 


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    TypeError
        if svector is complex
    OtherError
        when an other error
    """

    def __init__(self, svector, engine, verbose=False, **kwargs):
        # the measured vector obtained from experiment
        self.svector = np.asarray(svector, dtype=np.float)
        # dimensionality of measurement basis
        self.mdim = len(self.svector)
        # check for verbosity level
        self.verbose = verbose
        # store engine instance
        self.engine = engine
        # sensing flags
        self.flags = []

        # extract keyword arguments and allow overwrite of defaults
        self.opt_params = {"basis": 'random', "sparsity": 0.1,
                           "epsilon": 0.01, "freqs": [1e2, 1e3], "length": len(self.svector)}
        for key, value in kwargs.items():
            self.opt_params[key] = value

        # compute minimum number of measurements required to converge on
        # perfect reconstruction
        self.min_meas = np.ceil(2*self.opt_params["sparseness"]*np.log(
            self.mdim/self.opt_params["sparseness"]) + 5*self.opt_params["sparseness"]/4, )

        # default number of measurements to 10% over minimum number required (inaccurate for matched filtering)
        if "measurements" not in self.opt_params:
            self.opt_params["measurements"] = int(self.min_meas*1.1)

        # print number of measurements required
        if self.verbose:
            print("Minimum number of measurements for reconstruction: {}, Specified: {}".format(
                self.min_meas, self.opt_params["measurements"]))

        # generate measurement transform if not already defined using specified
        # basis
        if "transform" not in self.opt_params:
            if self.verbose:
                print("No transform specified, generating one with {} basis".format(
                    self.opt_params["basis"]))
            self.measure_gen()
        else:
            self.transform = self.opt_params["transform"]

        # add specified noise
        if self.opt_params["noise"]:
            self.svector += self.opt_params["noise"] * \
                np.random.rand(self.opt_params["length"], 1)
        else:
            # force noiseless convex optimisation
            self.opt_params["epsilon"] = 0.0

        # generate measurement vector with specified noise
        self.measurement = np.dot(self.transform, self.svector).T

    # generates desired measurement basis set with given parameters
    def measure_gen(self):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # pre-allocate transform matrix
        transform = np.zeros(
            (self.opt_params["measurements"], self.mdim), dtype=np.complex64)

        if self.opt_params["basis"] == "random":
            transform = 2*np.random.ranf(size=np.shape(transform))-1

        elif self.opt_params["basis"] == "fourier":
            # predefine measurement steps (we always assume a one second
            # measurement period)
            self.t = np.arange(
                0, self.opt_params["time"], 1/self.opt_params["fs"])

            # generate random frequencies or use those in freq list
            rand_flag = len(
                self.opt_params["freqs"]) < self.opt_params["measurements"]
            # create storage container for selected indices
            if not rand_flag:
                self.rand_ints = []

            self.opt_params["meas_freq"] = []
            print(rand_flag)
            for i in range(self.opt_params["measurements"]):
                if rand_flag:
                    # choose a random frequency over the given range
                    freq = (self.opt_params["freqs"][1] - self.opt_params["freqs"]
                            [0])*np.random.ranf()+self.opt_params["freqs"][0]
                else:
                    # choose a random frequency in the provided set and save index of chosen int
                    randint = np.random.randint(
                        low=0, high=len(self.opt_params["freqs"]))
                    # ensure frequency has not been chosen before (inefficient but in the scheme of things, unimportant)
                    while randint in self.rand_ints:
                        randint = np.random.randint(
                            low=0, high=len(self.opt_params["freqs"]))
                    # save chosen frequency
                    self.rand_ints.append(randint)
                    # add to set
                    freq = self.opt_params["freqs"][randint]
                # add frequency to selection
                self.opt_params["meas_freq"].append(freq)
                # np.imag(np.exp(-1j*2*np.pi*freq*self.t))
                transform[i, :] = -np.sin(2*np.pi*freq*self.t)

        else:
            print("unknown measurement basis specified: exiting ")
            os._exit(1)
        self.transform = transform

    # method to pass optimisation problem to CVX (will soon be changed to pycvx)
    def cvx_recon(self):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # set cvx flag
        self.flags.append("cvx_recon")

        if self.verbose:
            print("Performing optimisation using Matlab CVX")
        # convert to compatible data structure
        matlab_transform = matlab.double(
            self.transform.tolist(), is_complex=True)

        # generate noisy measured vector and convert to matlab type double
        if self.opt_params["noise"]:
            noisy = self.measurement.reshape([1, -1]) + self.opt_params[
                "noise"]*np.random.rand(self.opt_params["measurements"], 1)
            measurement = matlab.double(noisy.tolist(), is_complex=True)
        else:
            measurement = matlab.double(self.measurement.reshape(
                [-1, 1]).tolist(), is_complex=True)

        # compute reconstruction
        self.u_recon = np.asarray(self.engine.atomic_cvx(
            measurement, matlab_transform, self.opt_params["epsilon"]))

        # store error vector
        self.u_error = self.svector-self.u_recon
        # compute error using both l1 and l2 norm
        self.metrics = {'l1': np.linalg.norm(
            self.u_error, ord=1), "l2": np.linalg.norm(self.u_error, ord=2)}

    # find time index of most prominent spike using python optimisation
    def py_match(self, time, power=1):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # set single shot matched filter flag
        self.flags.append("matched")

        # set time range
        tau_range = np.arange(
            0, self.opt_params["time"], 1/self.opt_params["fs"])

        if self.verbose:
            print("Performing compressive matched filtering")

        # generate noisy measured vector and convert to matlab type double
        if self.opt_params["noise"]:
            self.measurement = (np.dot(self.transform, self.svector) +
                                self.opt_params["noise"]*np.random.rand(self.opt_params["measurements"], 1)).T
        else:
            self.opt_params["epsilon"] = 0.0
            self.measurement = np.dot(self.transform, self.svector).T

        # define autocorrelation function
        def tau_match(tau): return np.abs(self.measurement.dot(
            self.transform.dot(self._sig_shift(self.opt_params["template"], tau))))
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
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        self.flags.append('atomic')

        if self.verbose:
            print("Performing compressive matched filtering")

        # define nuke function
        self.nuke = np.ones((self.mdim, 1), dtype=float)
        # define autocorrelation function

        def tau_match(template, tau): return np.abs(np.real(self.measurement.dot(
            self.transform.dot(np.multiply(self.nuke, self._sig_shift(template, tau))))))

        spike = 0
        self.template_recon = np.zeros((self.mdim, 1))
        while spike < max_spikes:
            spike += 1

            # compute correlation function over parameter space
            self.correlation = np.zeros((self.mdim,), dtype=np.complex64)
            for step, tau in enumerate(np.arange(0, self.opt_params["time"], 1/self.opt_params["fs"])):
                test = self.transform.dot(np.multiply(
                    self.nuke, self._sig_shift(self.opt_params["template"], tau)))
                self.correlation[step] = tau_match(
                    self.opt_params["template"], tau)**power

            # extract maximum peak
            max_ind = np.argmax(a=self.correlation)
            # add peak to vector nuke
            self.nuke[max_ind-len(self.opt_params["template"])
                                  :max_ind+2*len(self.opt_params["template"])] = 0
            # subtract template from measurement vector
            #self.measurement -= self.transform.dot(self._sig_shift(self.opt_params["template"], max_ind/self.opt_params["fs"])).T
            self.template_recon += self._sig_shift(
                self.opt_params["template"], max_ind/self.opt_params["fs"]).reshape([-1, 1])

        # compute error
        self.m_error = self.svector-self.template_recon
        # compute errors
        self.metrics = {'l1': np.linalg.norm(
            self.m_error, ord=1), "l2": np.linalg.norm(self.m_error, ord=2)}

    # comptues an estimate for the amplitude given the best guess
    def amp_get(self, max_ind, template):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # should we use the full reconstruction or just the single spike?
        waveform = self._sig_shift(
            template, max_ind/self.opt_params["fs"]).reshape([-1, 1])
        # compute estimated amplitude
        t_waveform = self.transform.dot(waveform)
        print(np.shape(self.measurement))
        wave_amp = self.measurement.dot(
            t_waveform)/(np.linalg.norm(t_waveform, 2)**2)
        return wave_amp

    # shifts signal vector by tau
    def _sig_shift(self, template, tau):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # create zeroed vector (inefficent!)
        shift = np.zeros((self.mdim, 1))
        int_pos = int(np.round(tau*self.opt_params["fs"]))
        temp_len = len(template)
        # force clipping
        if int_pos < 0:
            int_pos = 0
        elif int_pos > self.mdim - temp_len:
            int_pos = int(self.mdim - temp_len)
        # shift vector by specified amount
        shift[int_pos: int_pos + temp_len] = template.reshape([-1, 1])

        return shift

    # basic plot code for the reconstructed signal
    def plot_recon(self, time):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        print(np.shape(time), np.shape(self.u_recon))
        plt.plot(time, self.svector, color="r", label='Original')
        plt.plot(time, self.u_recon, 'g', label='Reconstruction')
        plt.title("Samples: {}, Measurements: {}, Basis: {}, Noise Amplitude: {}, ||Ax - b||_2 = {:2e}".format(self.opt_params["length"],
                                                                                                               self.opt_params[
                                                                                                                   "measurements"],
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
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        fig, axx = plt.subplots(2, sharex=True)
        l1, = axx[0].plot(time, self.template_recon, 'g')
        axx[0].grid(True)
        axx[0].set_title("Notched matched filter - Samples: {}, Measurements: {}, Basis: {}, Noise Amplitude: {}, :L_2 Error: {}".format(self.opt_params["length"],
                                                                                                                                         self.opt_params[
                                                                                                                                             "measurements"],
                                                                                                                                         self.opt_params[
                                                                                                                                             "basis"],
                                                                                                                                         self.opt_params[
                                                                                                                                             "noise"],
                                                                                                                                         self.metrics["l2"]))
        l2, = axx[0].plot(time, self.svector, 'r')
        #plt.legend([l1,l2], ["Reconstructed", "Original"])
        l3, = axx[1].plot(time, self.nuke, 'b')
        axx[1].grid(True)
        axx[1].set_xlabel("Time (s)")
        plt.show()

    # stores an optimised object class in the archive
    def hdf_flush(self, name=None, group="today"):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        # get random seed of run
        self.opt_params["random_seed"] = np.random.get_state()[1][0]

        if name == None:
            name = "{}_{}_{}_{}".format(self.opt_params["basis"],
                                        self.opt_params["length"],
                                        self.opt_params["measurements"],
                                        self.opt_params["random_seed"])

        # store attributes
        attributes = [(key, val) for key, val in self.opt_params.items()]

        # flush all relevant data sets to group
        from archive_manager import data_flush
        # signal vector and key attributes
        data_flush(data=self.svector, group=group,
                   name="{}/raw_signal".format(name), attrs=attributes)
        # transform
        data_flush(data=self.transform, group=group,
                   name="{}/transform".format(name))
        # measurement vector
        data_flush(data=self.measurement, group=group,
                   name="{}/measurement".format(name))
        # cycle through flags
        if 'cvx_recon' in self.flags:
            # reconstructed signal
            data_flush(data=self.u_recon, group=group,
                       name="{}/u_recon".format(name))
        if 'atomic' in self.flags:
            # reconstructed matched filter
            data_flush(data=self.template_recon, group=group,
                       name="{}/matched_filter".format(name))


# define box function
def box(t, start, end):
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """

    if type(t) is not list or type(t) is not np.array:
        t = np.asarray(t)
    return (t > start) & (t < end)

# define sum function method


def sig_sum(t, terms):
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """
    return sum(f(t) for f in terms)


def pulse_gen(freq=1/2e-3, tau=[3.0], amp=1):
    """
    generates a multipulse magnetic field signal
    """

    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """

    # list to hold spike function
    terms = []

    for time in tau:
        # generate sin template vector
        terms.append(lambda t, tau=time: box(t, tau, tau + 1/freq)
                     * amp*np.sin(2*np.pi*freq*(t-tau)))

    signal = lambda t, funcs=terms: sig_sum(t, funcs)

    # return generated signal function, template used and time vector
    return signal
