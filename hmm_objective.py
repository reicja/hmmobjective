#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:09:07 2018

@author: jan
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student
from scipy.misc import logsumexp
from hmmlearn._hmmc import _forward, _backward
from swarm_optimization import ParticleSwarm


def normalize(x, axis=None):
    sum_ = np.sum(x, axis=axis)
    if axis and x.ndim > 1:
        sum_[sum_ == 0] = 1
        shape = list(x.shape)
        shape[axis] = 1
        sum_.shape = shape
    x /= sum_


def sample_hmm(startprob, transmat, rv, n_samples):
    n_states = len(startprob)
    obs = np.zeros(shape=(n_samples, 1))
    states = [np.random.choice(n_states, size=1, p=startprob)[0]]
    obs[0] = rv[states[0]].rvs()
    for n in range(1, n_samples):
        states.append(np.random.choice(n_states, size=1,
                                       p=transmat[states[-1], :])[0])
        obs[n] = rv[states[-1]].rvs()
    return np.array(obs), np.array(states)


class _HMMBase(object):
    def __init__(self, n_states):
        self.n_states = n_states
        self.dim = n_states * (1 + n_states)
        self._params = {}
        self.params = {}

    def _logprob_X(self, X, **kwargs):
        """Abstract method implements emission distribution."""
        pass

    def _parse_particle_params(self, particle):
        """Abstract method parse particles into emission distribution
        parameters."""
        pass

    def _repair_zeros(self, arr):
        arr[arr < 0] = 1e-5

    def _parse_particle(self, particle):
        """Helper method parse parameters from PSO particle."""
        startprob = particle[:self.n_states]
        transmat = particle[self.n_states:self.n_states * (1 + self.n_states)]
        self._parse_particle_params(particle[
                    self.n_states * (1 + self.n_states):])

        return startprob, transmat

    def _parse_final(self, particle):
        """Helper method saves final estimates."""
        startprob, transmat = self._parse_particle(particle)
        self.params['startprob'] = startprob
        self.params['transmat'] = transmat
        self._parse_final_params()
        return self.params

    def _shape_probs(self, startprob, transmat):
        # XXX:
        # purposedly exploit mutable type numpy.array in particle swarm optim
        self._repair_zeros(startprob)
        normalize(startprob)

        self._repair_zeros(transmat)
        transmat = transmat.reshape((self.n_states, self.n_states))
        normalize(transmat, axis=1)
        return startprob, transmat

    def _compute_logprob(self, X, startprob, transmat, **kwargs):
        logprobX = self._logprob_X(X, **kwargs)
        n_samples, n_components = logprobX.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _forward(n_samples, n_components,
                 np.log(startprob),
                 np.log(transmat),
                 logprobX, fwdlattice)
        return logsumexp(fwdlattice[-1])

    def _cost(self, X, particle):
        """Method implements hmm cost function in the form of negative
        log-likelihood, which is to be minimized."""
        startprob, transmat = self._parse_particle(particle)
        startprob, transmat = self._shape_probs(startprob, transmat)
        return -self._compute_logprob(X, startprob, transmat, **self._params)

    def fit(self, X, lb, ub, n_particles=50, n_iters=1000, vmax=None):
        """Method fits the model with particle swarm optimization."""
        if vmax is None:
            vmax = .2 * np.ones(self.dim)
        else:
            assert(len(vmax) == self.dim)

        lb_ = np.zeros(self.n_states * (self.n_states + 1))
        lb = np.hstack((lb_, lb))

        ub_ = np.ones(self.n_states * (self.n_states + 1))
        ub = np.hstack((ub_, ub))

        assert(len(lb) == self.dim)
        assert(len(ub) == self.dim)

        cost_func = lambda prtcl: self._cost(X, prtcl)

        ps = ParticleSwarm(cost_func,
                           dim=self.dim,
                           n_particles=50,
                           n_iters=1000,
                           vmax=vmax,
                           lb=lb,
                           ub=ub)
        ps.run()

        params, llik = ps.get_solution()
        self._parse_final(params)

        print "Optimization terminated with loglik value of {0}".format(-llik)
        return -llik


class GHMM(_HMMBase):
    def __init__(self, n_states):
        super(GHMM, self).__init__(n_states)
        self._params['loc'] = []
        self._params['scale'] = []
        self.dim += 2 * n_states  # holds for 1D emissions

    def _logprob_X(self, X, **kwargs):
            return norm(**kwargs).logpdf(X)

    def _parse_particle_params(self, particle):
        """Parse parameters from particle in order. The order is
        (loc, scale)"""
        self._params['loc'] = particle[:self.n_states]
        sigma = particle[self.n_states:]
        self._repair_zeros(sigma)
        self._params['scale'] = sigma

    def _parse_final_params(self):
        self.params['loc'] = self._params['loc']
        self.params['scale'] = self._params['scale']


class THMM(_HMMBase):
    def __init__(self, n_states):
        super(THMM, self).__init__(n_states)
        self._params['df'] = []
        self._params['loc'] = []
        self._params['scale'] = []
        self.dim += 3 * n_states  # holds for 1D emissions

    def _logprob_X(self, X, **kwargs):
        return student(**kwargs).logpdf(X)

    def _parse_particle_params(self, particle):
        """Parse parameters from particle in order. The order is
        (df, loc, scale)"""
        df = particle[:self.n_states]
        self._repair_zeros(df)
        self._params['df'] = df
        self._params['loc'] = particle[self.n_states:2*self.n_states]
        sigma = particle[2*self.n_states:]
        self._repair_zeros(sigma)
        self._params['scale'] = sigma

    def _parse_final_params(self):
        self.params['df'] = self._params['df']
        self.params['loc'] = self._params['loc']
        self.params['scale'] = self._params['scale']


# TODO:

class ARHMM(_HMMBase):
    def __init__(self, X, n_states, order=1):
        super(ARHMM, self).__init__(X, n_states)
        self._params['phi'] = []
        self._params['loc'] = []
        self._params['scale'] = []
        self.dim += n_states * (2 + order)  # holds for 1D emissions
        self._p = order

    def _logprob_X(self, **kwargs):
        Xp = [self.X[:-self._p + i] for i in range(self._p)]
        Xp = np.hstack(Xp)
        loc = np.dot(kwargs['phi'], Xp)
        loc += kwargs['loc']
        scale = kwargs['scale']
        return norm(loc=loc, scale=scale).logpdf(self.X[self._p:])

    def _parse_particle_params(self, particle):
        sigma = particle[:self.n_states]
        self._repair_zeros(sigma)
        self._params['scale'] = sigma
        self._params['loc'] = particle[self.n_states:2*self.n_states]
        phi = particle[2*self.n_states:]
        self._params['phi'] = phi

    def _parse_final_params(self):
        self.params['phi'] = self._params['phi']
        self.params['scale'] = self._params['scale']
        self.params['loc'] = self._params['loc']
