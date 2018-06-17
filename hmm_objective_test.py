#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:46:38 2018

@author: jan
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student
from hmm_objective import GHMM, THMM
from hmm_objective import sample_hmm
from scipy import signal
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')


norm_pdfs = lambda pars, n_states: [norm(loc=pars['loc'][k],
                                         scale=pars['scale'][k])
                                    for k in range(n_states)]

t_pdfs = lambda pars, n_states: [student(loc=pars['loc'][k],
                                         scale=pars['scale'][k],
                                         df=pars['df'][k])
                                 for k in range(n_states)]

n_samples = 800
prior = np.array([.1, .85, .05])

a11, a12, a13 = .75, .2, .05
a21, a22, a23 = .07, .9, .03
a31, a32, a33 = .05, .05, .9
transmat = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

params_norm = {}
params_norm['loc'] = [-.1, .05, .9]
params_norm['scale'] = [1.5, .5, 1]

params_t = {}
params_t['loc'] = [-.1, .05, .9]
params_t['scale'] = [1.5, .5, 1]
params_t['df'] = [9, 15, 20]

obs, states = sample_hmm(startprob=prior,
                         transmat=transmat,
                         rv=norm_pdfs(params_norm, 3),
                         n_samples=n_samples)

hmm = GHMM(n_states=3)
# parameters clip
# loc, scale
lb = np.array([-1, -1, -1, 0, 0, 0])
ub = np.array([1, 1, 1, 3, 3, 3])
hmm.fit(obs, lb, ub)
print "Simulated data likelihood: -{0}".format(
        hmm._cost(obs, np.hstack((prior, transmat.flatten(),
                  params_norm['loc'],
                  params_norm['scale']))))

obs, states = sample_hmm(startprob=prior,
                         transmat=transmat,
                         rv=t_pdfs(params_t, 3),
                         n_samples=n_samples)

hmm = THMM(n_states=3)
# parameters clip
# df, loc, scale
lb = np.array([0, 0, 0, -5, -5, -5, 0, 0, 0])
ub = np.array([25, 25, 25, 5, 5, 5, 5, 5, 5])
hmm.fit(obs, lb, ub)
print "Simulated data likelihood: -{0}".format(
        hmm._cost(obs, np.hstack((prior, transmat.flatten(),
                  params_t['df'],
                  params_t['loc'],
                  params_t['scale']))))


# wawelet analysis
t = np.arange(len(obs))
sig = obs.flatten()
widths = np.arange(1, 50)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 1, 50], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()
