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
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000)) for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000))) for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
    },
    "STAT-1-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000)) for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000))) for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
    },
    "STAT-2": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000)) for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000))) for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
    },
    "STAT-2-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000)) for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), -1000, 1000))) for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
    },
}

starting_points = {
    "STAT-1": {
        "default": [[.6, 1]],
        "rounded": [],
    },
    "STAT-1-ni": {
        "default": [[.6, .05, 1]],
        "rounded": [], 
    },
    "STAT-2": {
        "default": [[.8, 1], [.2, 1]],
        "rounded": [],
    },
    "STAT-2-ni": {
        "default": [[.7, 0.01, 1], [.1, 0.01, 1]],
        "rounded": [],
    },
}

for key in problems:
    T, s = problems[key]['rounded'][1].transformation, problems[key]['rounded'][1].shift
    starting_points[key]['rounded'] = [np.linalg.solve(T, x - s) for x in starting_points[key]['default']]
    print(starting_points[key]['rounded'])

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


m, l = 0, 1
a, b = -5, 3

def get_stepsize_grid(_m, _l, _a=a, _b=b):
    print(_m, _l, _a, _b)
    return 10**np.hstack([
        np.linspace(_a, _m-_l, int(coarse * (_m-_l-_a) + 1))[:-1],
        np.linspace(_m-_l, _m+_l, int(fine * 2* _l + 1))[:-1], 
        np.linspace(_m+_l, _b, int(coarse * (_b-_m-_l) + 1))
    ])    
        
# m, l, a, b
grid_params = {
    ('STAT-1', 'BallWalk'): (-1, 1),
    ('STAT-1', 'CSmMALA'): (.5, 1.5),
    ('STAT-1', 'DikinWalk'): (-2, 2),
    ('STAT-1', 'Gaussian'): (-1, 1),
    ('STAT-1', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-1', 'Rounding BallWalk'): (-4, 1),
    ('STAT-1', 'Rounding Gaussian'): (-4, 1),
    ('STAT-1', 'Rounding Gaussian\nHit-And-Run'): (-4, 1),
    ('STAT-1-ni', 'BallWalk'): (-1, 1),
    ('STAT-1-ni', 'CSmMALA'): (-1, 1),
    ('STAT-1-ni', 'DikinWalk'): (-1, 1),
    ('STAT-1-ni', 'Gaussian'): (-1.5, 1),
    ('STAT-1-ni', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-1-ni', 'Rounding BallWalk'): (-4, 1),
    ('STAT-1-ni', 'Rounding Gaussian'): (-4, 1),
    ('STAT-1-ni', 'Rounding Gaussian\nHit-And-Run'): (-4, 1),
    ('STAT-2', 'BallWalk'): (-.5, 1.5),
    ('STAT-2', 'CSmMALA'): (.5, 1.5),
    ('STAT-2', 'DikinWalk'): (0, 2),
    ('STAT-2', 'Gaussian'): (0, 1.5),
    ('STAT-2', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-2', 'Rounding BallWalk'): (-1, 1),
    ('STAT-2', 'Rounding Gaussian'): (-1, 1),
    ('STAT-2', 'Rounding Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-2-ni', 'BallWalk'): (-.5, 1.5),
    ('STAT-2-ni', 'CSmMALA'): (-1, 1),
    ('STAT-2-ni', 'DikinWalk'): (-1, 1),
    ('STAT-2-ni', 'Gaussian'): (-1, 1),
    ('STAT-2-ni', 'Gaussian\nHit-And-Run'): (-1, 1),
    ('STAT-2-ni', 'Rounding BallWalk'): (0, 1),
    ('STAT-2-ni', 'Rounding Gaussian'): (0, 1),
    ('STAT-2-ni', 'Rounding Gaussian\nHit-And-Run'): (0, 1),
}

stepsize_grids = {(problem, proposal): get_stepsize_grid(*grid_params[(problem, proposal)]) if (problem, proposal) in grid_params else get_stepsize_grid(-1, 1) for problem in problems for proposal in proposals}

targets = {
    "neff": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states))]*n_chains),
    "neff/t": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states)) / (elapsed / states.shape[1])]*n_chains),
    "rhat": (lambda rhat, elapsed, accrate, states: [rhat]*n_chains),
    "acc": (lambda rhat, elapsed, accrate, states: np.mean(accrate, axis=0)),
    "esjd": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1)),
    "esjd/t": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1) / (elapsed / states.shape[1])),
    "t": (lambda rhat, elapsed, accrate, states: [elapsed / states.shape[1]]*n_chains),
}

target_display_names = {
    "neff":   r"min $n_{\mathrm{eff}}$",
    "neff/t": r"min $n_{\mathrm{eff}}/t$",
    "rhat":   r"max $\hat{R}$",
    "acc":    "Acceptance rate",
    "esjd":   "ESJD",
    "esjd/t": "$\mathrm{ESJD}/t$",
    "t":      "$T/n$",
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


opt_sampling = {'STAT-1': ('CSmMALA', 4.39397056076079), 'STAT-1-ni': ('Gaussian\nHit-And-Run', 0.1), 'STAT-2': ('Gaussian\nHit-And-Run', 0.1), 'STAT-2-ni': ('Gaussian\nHit-And-Run', 1)}

def posterior_sampling(problem, Proposal, stepsize):
    mcs = [hopsy.MarkovChain(problem, hopsy.UniformHitAndRunProposal) for i in range(n_chains)]
    for mc in mcs: mc.proposal.stepsize = stepsize
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


def bruteforce_sampling(Proposal, problem, dim, starting_points, stepsize, seed):
    """
        Samples problem using a proposal of type Proposal from starting_point and with
        stepsize.
    """
    mcs = [hopsy.MarkovChain(problem, Proposal, starting_points[i]) for i in range(n_chains)]
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
args_idx = []
args_key = []

for problem_key in problems:
    n_jobs = len(args)
    _problems = problems[problem_key]
    for proposal_key, Proposal in proposals.items():
        if rounding[proposal_key]:
            variant = "rounded"
        else:
            variant = "default"
            
        dim, problem = _problems[variant]
            
        for stepsize in stepsize_grids[(problem_key, proposal_key)]:
            for seed in range(n_seeds):
                rng = hopsy.RandomNumberGenerator(seed, n_chains)
                uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
                draws = [int(uniform(rng)) for i in range(n_chains)]
                
                _starting_points = [starting_points[problem_key][variant][i] for i in draws]
                
                args += [(Proposal, problem, dim, _starting_points, stepsize, seed)]
    n_jobs = len(args) - n_jobs
    args_key += [problem_key] * n_jobs # all these new elements belong to problem_key
    
args_idx += [(problem, proposal, i, seed) for problem in problems
                                          for proposal in proposals
                                          for i, _ in enumerate(stepsize_grids[(problem, proposal)])
                                          for seed in range(n_seeds)]
