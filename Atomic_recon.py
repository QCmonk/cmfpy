


from capy import *
from qChain import *
from operators import *

import numpy as np 
import matplotlib.pyplot as plt

def recon_pulse():
    np.random.seed(141) #  seed 131 fails with 50 measurements
    # generate sparse source signal to get template 

    # compute full measurement record in the frequency range f_range - gives Fourier Sine coefficients
    freqs, projs, signal = pseudo_fourier(struct="pulse", sig_amp=1,sig_freq=1e4, f_range=[9e3,11e3,2], tau=[1e-2], noise=0.00, plot=False, verbose=False)        


    
    # define user variables for compressive sensing
    user_vars = {"engine": matlab.engine.start_matlab(),
                 "sparseness":       400, 
                 "measurements":     100,
                 "epsilon":         0.02,
                 "basis":      "fourier",
                 "freqs":          freqs,
                 "noise":           0.00,
                 "time":             0.1,
                 "fs":               1e5}


    time = np.arange(0, user_vars["time"], 1/user_vars["fs"])
    plt.plot(signal(time))

    comp = CAOptimise(svector=signal(time).reshape([-1,1]), verbose=True, **user_vars)
    # reassign measurement projectors, selecting only those chosen for reconstruction
    
    comp.measurement = projs[[i for i in comp.rand_ints]] - 0.5 # (projs[comp.rand_ints]-0.5)/(np.max(projs[comp.rand_ints] -0.5))

    comp.cvx_recon()

    #comp.py_notch_match(power=1, max_spikes=1)
    #comp.notch_match_plot(time)

    #plt.plot(comp.opt_params["meas_freq"], comp.measurement/np.max(comp.measurement), '*', label="Sensor")
    #plt.plot(comp.opt_params["meas_freq"],np.dot(comp.transform, signal(time))/np.max(np.dot(comp.transform, signal(time))), '*', label="Exact")
    # plt.legend()
    # plt.show()
    comp_f = comp.opt_params["meas_freq"]
    comp_p = comp.measurement + 0.5
    recon = comp.u_recon
    plot_gen_1(freqs, projs, time, signal)
    plot_gen_2(freqs, projs, comp_f, comp_p, time, recon, measurements=len(comp.measurement))

def checking():
    # time of pulse event (single)
    tau = 2e-2
    # frequncy of desired neural signal
    sig_freq = 1e4
    # amplitude of signal in Hz*gyro
    sig_amp = 1
    params = {"struct": ["pulse", "sinusoid", "constant"], 
                      "freqb": [sig_freq, 50, 0],          # frequency in Hz
                      "tau": [[tau], None, None],          # time event of pulse
                      "amp":[sig_amp/gyro, 10/gyro, sig_freq/gyro], # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                      "misc": [None,None,None]}          # misc parameters
    # define hamiltonian class
    ham = Hamiltonian()
    # create specified signal field
    fields = field_gen(field_params=params)
    # plot field
    field_plot(fields)
    # initialise system
    atom = SpinSystem(init="super")
    # generate Hamiltonian defining system evolution
    ham.generate_field_hamiltonian(fields)
    time, probs, pnts = atom.state_evolve(t=[0,0.2,1/1e5], 
                                          hamiltonian=ham.hamiltonian, 
                                          project=meas1["1"], 
                                          bloch=[False, 20])
    # plot evolution on bloch sphere
    #atom.bloch_plot(points=pnts)
    # plot probability distribution
    atom.prob_plot(time, probs)

if __name__ == "__main__":
    params = {"struct": ["sinusoid", "constant", "constant"], 
                  "freqb": [0, 1, 1],          # frequency in Hz
                  "tau": [None, None, None],          # time event of pulse
                  "amp":[0, 10/gyro, 10/gyro], # amplitude in Gauss -> 1 Gauss ~= 700000 Hz precession
                  "misc": [None,None,None]}          # misc parameters

    # define hamiltonian class
    ham = Hamiltonian()
    # create specified signal field
    fields = field_gen(field_params=params)
    # plot field
    #field_plot(fields)
    # initialise system
    atom = SpinSystem(init="super")
    # generate Hamiltonian defining system evolution
    ham.generate_field_hamiltonian(fields)
    time, probs, pnts = atom.state_evolve(t=[0,0.1,1/1e4], 
                                          hamiltonian=ham.hamiltonian, 
                                          project=meas1["1"], 
                                          bloch=[True, 10])
    atom.bloch_plot(pnts)


