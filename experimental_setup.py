import hopsy
import numpy as np
import arviz as az

import x3c2fluxpy as x3c
#import x3cflux2adapter as _x3c

import time

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

import warnings
from multiprocessing import Pool

import dill
from tqdm.notebook import tqdm

import corner

session = "bruteforce"

dims = [2]#, 5, 10, 20]#, 30, 50]

n_parallel = 50
n_seeds = 5
n_chains = 10

n_samples = 10_000
n_samples_order = 1

rhat_threshold=1.05
N_max = 20

fine, coarse = 16, 5 

problems = {
    "STAT-1": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-1") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000)), "STAT-1") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
    },
    "STAT-1-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-1-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000)), "STAT-1-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
    },
    "STAT-2": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-2") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000)), "STAT-2") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
    },
    "STAT-2-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-2-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000)), "STAT-2-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
    },
}

proposals = {
#    "Adaptive Metropolis": hopsy.AdaptiveMetropolisProposal,
    "BallWalk": hopsy.BallWalkProposal,
    "CSmMALA": hopsy.CSmMALAProposal,
    "DikinWalk": hopsy.DikinWalkProposal,
    "Gaussian": hopsy.GaussianProposal,
    "Gaussian\nHit-And-Run": hopsy.GaussianHitAndRunProposal,
    "Rounding BallWalk": hopsy.BallWalkProposal,
    "Rounding Gaussian": hopsy.GaussianProposal,
    "Rounding Gaussian\nHit-And-Run": hopsy.GaussianHitAndRunProposal,
}

# last 3 algorithms will use rounding
rounding = {
    proposal: (i >= len(proposals) - 3) for i, proposal in enumerate(proposals)
}

def get_stepsize_grid(m, l, a = -5, b = 3):
    m, l = 0, 1
    a, b = -5, 3
    return np.hstack([
        np.linspace(a, m-l, coarse * (m-l-a) + 1)[:-1],
        np.linspace(m-l, m+l, fine * 2* l + 1)[:-1], 
        np.linspace(m+l, b, coarse * (b-m-l) + 1)
    ])    
        
# m, l, a, b
grid_params = {
    ('STAT-1', 'BallWalk'): (-1, 1),
    ('STAT-1', 'CSmMALA'): (.5, 1.5),
    ('STAT-1', 'DikinWalk'): (-2, 2),
    ('STAT-1', 'Gaussian'): (-1, 1),
    ('STAT-1', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-1-ni', 'BallWalk'): (-1, 1),
    ('STAT-1-ni', 'CSmMALA'): (-1, 1),
    ('STAT-1-ni', 'DikinWalk'): (-1, 1),
    ('STAT-1-ni', 'Gaussian'): (-1.5, 1),
    ('STAT-1-ni', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-2', 'BallWalk'): (-.5, 1.5),
    ('STAT-2', 'CSmMALA'): (.5, 1,5),
    ('STAT-2', 'DikinWalk'): (0, 2),
    ('STAT-2', 'Gaussian'): (0, 1.5),
    ('STAT-2', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-2-ni', 'BallWalk'): (-.5, 1.5),
    ('STAT-2-ni', 'CSmMALA'): (-1, 1),
    ('STAT-2-ni', 'DikinWalk'): (-1, 1),
    ('STAT-2-ni', 'Gaussian'): (-1, 1),
    ('STAT-2-ni', 'Gaussian\nHit-And-Run'): (-1, 1),
}

stepsize_grids = {(problem, proposal): get_stepsize_grid(*grid_params[(problem, proposal)]) for problem in problems for proposal in proposals}

targets = {
    r"min $n_{\mathrm{eff}}$": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states))]*n_chains),
    r"min $n_{\mathrm{eff}}$/t": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states)) / (elapsed / states.shape[1])]*n_chains),
    r"max $\hat{R}$": (lambda rhat, elapsed, accrate, states: [rhat]*n_chains),
    "Acceptance rate": (lambda rhat, elapsed, accrate, states: np.mean(accrate, axis=0)),
    "ESJD": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1)),
    "ESJD/t": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1) / (elapsed / states.shape[1])),
    "T/n": (lambda rhat, elapsed, accrate, states: [elapsed / states.shape[1]]*n_chains),
}

def uniform_sampling(problem):
    mcs = [hopsy.MarkovChain(problem, hopsy.UniformHitAndRunProposal) for i in range(n_chains)]
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    accrate, states = hopsy.sample(mcs, rngs, n_samples, dim)
    rhat = np.max(hopsy.rhat(states))
    
    i = 0
    while not(rhat < rhat_threshold) and i < n_max:
        _accrate, _states = hopsy.sample(mcs, rngs, n_samples, dim)
        
        states = np.concatenate([states, _states], axis=1)
        rhat = np.max(hopsy.rhat(states))
        
        i += 1
        
    return states


def bruteforce_sampling(Proposal, problem, starting_point, dim, stepsize, seed):
    """
        Samples problem using a proposal of type Proposal from starting_point and with
        stepsize.
    """
    mcs = [hopsy.MarkovChain(problem, Proposal, starting_point) for i in range(n_chains)]
    for mc in mcs: mc.proposal.stepsize = stepsize
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    elapsed = time.time()
    accrate, states = hopsy.sample(mcs, rngs, n_samples, dim)
    elapsed = time.time() - elapsed
    rhat = np.max(hopsy.rhat(states))
    
    accrate = [accrate]
    
    i = 0
    while not(rhat < rhat_threshold) and i < N_max:
        elapsed = time.time() - elapsed
        _accrate, _states = hopsy.sample(mcs, rngs, n_samples, dim)
        elapsed = time.time() - elapsed
        
        accrate += [_accrate]
        states = np.concatenate([states, _states], axis=1)
        
        rhat = np.max(hopsy.rhat(states))
        
        i += 1
        
    result = []
    for _, target in targets.items():
        result.append(target(rhat, elapsed/n_chains, np.array(accrate), states))
        
    return result


args = []
args_idx = {problem_key: [] for problem_key in problems}
args_key = []

for problem_key in problems:
    n_jobs = len(args)
    args += [(Proposal, problem, dim, stepsize, seed)   for _, Proposal in proposals.items()
                                                        for dim, problem, _ in problems[problem_key] 
                                                        for stepsize in stepsize_grid
                                                        for seed in range(n_seeds)
            ]
    n_jobs = len(args) - n_jobs
    args_key += [problem_key] * n_jobs # all these new elements belong to problem_key
    
for problem_key in problems:
    args_idx[problem_key] += [(i, j, k, seed)   for i, _ in enumerate(proposals)
                                                for j, _ in enumerate(problems[problem_key])
                                                for k, _ in enumerate(stepsize_grid)
                                                for seed in range(n_seeds)]
