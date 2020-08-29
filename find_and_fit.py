#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import P4J
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.gridspec as gridspec


# # Load a data file 
data = np.loadtxt("datafile.dat", usecols=(0,1,2), skiprows=1)
hjd = data[:,0]
mag = data[:,1]
err = data[:,2]


# # Define the Fourier series function
def fourier_series(x, f, n=0):
#    """
#    Returns a symbolic fourier series of order `n`.
#
#    :param n: Order of the fourier series.
#    :param x: Independent variable
#    :param f: Frequency of the fourier series
#    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


# # Finding the period and plotting it with a Fourier fit.
my_per = P4J.periodogram(method='MHAOV') 
# Multi-harmonic Analysis of Variance

my_per.set_data(hjd, mag, err)
my_per.frequency_grid_evaluation(fmin=0.0, fmax=0.5, fresolution=1e-4)  
# frequency sweep parameters
my_per.finetune_best_frequencies(fresolution=1e-4, n_local_optima=10)
freq, per = my_per.get_periodogram()
fbest, pbest = my_per.get_best_frequencies() 
# Return best n_local_optima frequencies

f,ax = plt.subplots(2,2,figsize=(24,18))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], hspace=0.2,wspace=0.2)

#################
f0_ax1 = plt.subplot(gs[0])
f0_ax1.plot(freq, per)
ymin, ymax = f0_ax1.get_ylim()
f0_ax1.plot([fbest[0], fbest[0]], [ymin, ymax], linewidth=8, alpha=0.2)
f0_ax1.set_ylim([ymin, ymax])
#f0_ax1.set_xlim(0,0.3)
f0_ax1.set_xlabel('Frequency [1/HJD]', fontsize=15)
f0_ax1.set_ylabel('MHAOV Periodogram', fontsize=15)
f0_ax1.set_title('Periodogram', fontsize=25)
f0_ax1.grid()

#################
f0_ax2 = plt.subplot(gs[1])
phase = np.mod(hjd, 1.0/fbest[0])*fbest[0]
phase_con=np.concatenate([np.sort(phase), np.sort(phase)+1.0])
idx = np.argsort(phase)
idx_con = np.concatenate([np.sort(phase_con), np.sort(phase_con)+1.0])
mag_con = np.concatenate([mag[idx], mag[idx]])
f0_ax2.scatter(phase_con, mag_con)
f0_ax2.set_title('Folded Light Curve', fontsize=25)
f0_ax2.set_xlabel('Phase @ %0.5f [1/d], %0.5f [d]' %(fbest[0], 1.0/fbest[0]), fontsize=15)
f0_ax2.set_ylabel('Magnitude', fontsize=15)
#f0_ax2.set_ylim(-1.28, -0.98)
#f0_ax2.legend()
f0_ax2.grid()

####################
f1_ax1 = plt.subplot(gs[2])
f1_ax1.scatter(hjd, mag)
f1_ax1.set_xlabel('Observation Date (HJD)', fontsize=15)
f1_ax1.set_ylabel('Magnitude', fontsize=15)
f1_ax1.set_title('Observed Light Curve', fontsize=25)
f1_ax1.grid()
#f1_ax1.set_ylim(-1.28, -0.98)


####################
f1_ax2 = plt.subplot(gs[3])
x, y = variables('x, y')
w, = parameters('fbest')
model_dict = {y: fourier_series(x, f=w, n=9)}
print(model_dict)
# Make step function data
xdata = phase_con
ydata = mag_con

# Define a Fit object for this model and data
fit = Fit(model_dict, x=xdata, y=ydata)
fit_result = fit.execute()
print(fit_result)

# Plot the result
f1_ax2.scatter(xdata, ydata)
f1_ax2.plot(xdata, fit.model(x=xdata, **fit_result.params).y, 
         lw=5, alpha=0.3, c='crimson')
f1_ax2.set_xlabel('Phase', fontsize=15)
f1_ax2.set_ylabel('Magnitude', fontsize=15)
f1_ax2.set_title('Fourier Fit of the Phase Folded Light Curve', fontsize=25)
f1_ax2.grid()
#f1_ax2.set_ylim(-1.28, -0.98)



plt.savefig('output_example.pdf')





