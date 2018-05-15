#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:46:38 2018

@author: jan
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student
from swarm_optimization import ParticleSwarm
from hmm_objective import GaussianHMMCost, StudentHMMCost


def gen_hmm_obs(pi, A, rv, n_samples):
    n_states = len(pi)
    obs = np.zeros(shape=(n_samples, 1))
    states = [np.random.choice(n_states, size=1, p=pi)[0]]
    obs[0] = rv[states[0]].rvs()
    for n in range(1, n_samples):
        states.append(np.random.choice(n_states, size=1,
                                       p=A[states[-1], :])[0])
        obs[n] = rv[states[-1]].rvs()

    return np.array(obs), np.array(states)



norm_pdfs = lambda pars: [norm(loc=pars['mean1'], scale=pars['sig1']),
                          norm(loc=pars['mean2'], scale=pars['sig2'])]

t_pdfs = lambda pars: [student(loc=pars['mean1'], scale=pars['sig1'],
                               df=pars['df1']),
                       student(loc=pars['mean2'], scale=pars['sig2'],
                               df=pars['df2'])]

n_samples = 100
prior = np.array([.15, .85])

a11 = .75
a12 = .25
a21 = .07
a22 = .93
transmat = np.array([[a11, a12], [a21, a22]])

params = {}
params['mean1'] = -.1
params['mean2'] = .05

params['sig1'] = 1.5
params['sig2'] = .5

params['df1'] = 9
params['df2'] = 15

obs, states = gen_hmm_obs(pi=prior,
                          A=transmat,
                          rv=norm_pdfs(params),
                          n_samples=n_samples)


hmm = GaussianHMMCost(obs, n_states=2)

ps = ParticleSwarm(hmm.cost,
                   dim=hmm.dim,
                   n_particles=50,
                   n_iters=250,
                   vmax=np.array([.2, .2, .2, .2, .2, .2, 1, 1, 1, 1]),
                   lb=np.array([-1, -1, -1, -1, -1, -1, -1.5, -1.5, 0, 0]),
                   ub=np.array([1, 1, 1, 1, 1, 1, 1.5, 1.5, 2, 2]))

ps.run()

print "Particle Swarm solution: "
for key, val in zip(hmm._parse_final(ps.get_solution()[0]).keys(),
                    hmm._parse_final(ps.get_solution()[0]).values()):
    print key, val

print "\n"
print "Particle Swarm solution likelihood: {0}".format(ps.get_solution()[1])
print "\n"
print "Simulated data likelihood: {0}".format(
        hmm.cost(np.hstack((prior, transmat.flatten(),
                            params['mean1'], params['mean2'],
                            params['sig1'], params['sig2']))))


obs, states = gen_hmm_obs(pi=prior,
                          A=transmat,
                          rv=t_pdfs(params),
                          n_samples=n_samples)


hmm = StudentHMMCost(obs, n_states=2)

ps = ParticleSwarm(hmm.cost,
                   dim=hmm.dim,
                   n_particles=100,
                   n_iters=1000,
                   vmax=np.array([.2, .2, .2, .2, .2, .2, 2, 2, 2, 2, 2, 2]),
                   lb=np.array([0, 0, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0]),
                   ub=np.array([1, 1, 1, 1, 1, 1, 25, 25, 5, 5, 5, 5]))

ps.run()

print "Particle Swarm solution: "
for key, val in zip(hmm._parse_final(ps.get_solution()[0]).keys(),
                    hmm._parse_final(ps.get_solution()[0]).values()):
    print key, val

print "\n"
print "Particle Swarm solution likelihood: {0}".format(ps.get_solution()[1])
print "\n"
print "Simulated data likelihood: {0}".format(
        hmm.cost(np.hstack((prior, transmat.flatten(),
                            params['df1'], params['df2'],
                            params['mean1'], params['mean2'],
                            params['sig1'], params['sig2']))))
