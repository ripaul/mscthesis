import hopsy
import numpy as np
import arviz as az

import x3c2fluxpy as x3c
#import x3cflux2adapter as _x3c

import time

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
from multiprocessing import Pool

import dill
from tqdm.notebook import tqdm

import corner

session = "bruteforce"

n_parallel = 50
n_seeds = 5
n_chains = 10

n_test_samples = 100
n_samples_order = 1

stepsize_grid = 10**np.array(np.hstack([np.linspace(-5, -2, 15)[:-1], np.linspace(-2, 0, 41)[:-1], np.linspace(0, 3, 15)]))

problems = {
    #"Truncated Gaussian": [(dim, hopsy.add_box_constraints(
    #    hopsy.Problem(-np.identity(dim), [0]*dim, hopsy.Gaussian(mean=np.zeros(dim)), starting_point=1e-5*np.ones(dim)), 0, 100), "") for dim in dims],
    "STAT-1": [(model.A.shape[1], hopsy.Problem(model.A, model.b, model, model.initial_point), "STAT-1") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]], #+
    "STAT-1-ni": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-1-ni") 
                  for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]], #+
    "STAT-2": [(model.A.shape[1], hopsy.Problem(model.A, model.b, model, model.initial_point), "STAT-2") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]], #+
    "STAT-2-ni": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-2-ni") 
                  for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]], #+
}

proposals = {
#    "Adaptive Metropolis": hopsy.AdaptiveMetropolisProposal,
    "BallWalk": hopsy.BallWalkProposal,
    "CSmMALA": hopsy.CSmMALAProposal,
    "DikinWalk": hopsy.DikinWalkProposal,
    "Gaussian": hopsy.GaussianProposal,
    "Gaussian Hit-And-Run": hopsy.GaussianHitAndRunProposal,
}

targets = {
    r"$n_{\mathrm{eff}}$": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states))]*n_chains),
    r"$n_{\mathrm{eff}}$/t": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states)) / (elapsed / states.shape[1])]*n_chains),
    r"$\hat{R}$": (lambda rhat, elapsed, accrate, states: [rhat]*n_chains),
    "Acceptance Rate": (lambda rhat, elapsed, accrate, states: np.mean(accrate, axis=0)),
    "ESJD": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1)),
    "ESJD/t": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1) / (elapsed / states.shape[1])),
    "T/n": (lambda rhat, elapsed, accrate, states: [elapsed / states.shape[1]]*n_chains),
}

rhat_threshold=1.05
n_max = 20

def f(Proposal, problem, dim, stepsize, seed):
    mcs = [hopsy.MarkovChain(problem, Proposal) for i in range(n_chains)]
    for mc in mcs: mc.proposal.stepsize = stepsize
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    elapsed = time.time()
    accrate, states = hopsy.sample(mcs, rngs, 100000)
    elapsed = time.time() - elapsed
    rhat = np.max(hopsy.rhat(states))
    
    accrate = [accrate]
    
    i = 0
    while not(rhat < rhat_threshold) and i < n_max:
        elapsed = time.time() - elapsed
        _accrate, _states = hopsy.sample(mcs, rngs, 10000)
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

    
if n_parallel > 1:
    with Pool(n_parallel) as p:
        data_arr = p.starmap(f, args)
else:
    data_arr = []
    for arg in args:
        data_arr.append(f(*arg))

with open(session + '_data', "wb") as fhandle:
    dill.dump(data_arr, fhandle)
    #!git add * && git commit -m "bruteforce run"