import hopsy
import numpy as np
import arviz as az

import x3c2fluxpy as x3c2
#import x3cflux2adapter as _x3c

import time

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import cm

import warnings
from multiprocessing import Pool

import dill
from tqdm.notebook import tqdm


dims = [2]#, 5, 10, 20]#, 30, 50]

n_parallel = 60
n_seeds = 1#5
n_chains = 10
#n_chains = 2

n_samples = 10_000
n_samples_order = 1

rhat_threshold=1.05
N_max = 50
#N_max = 10

keep_n_samples = 500

lb, ub = -1000, 1000
lb_ni, ub_ni = -1000, 1000

dim = 20
A, b = [[1] + [0] * (dim-1)], [ub]
gauss = hopsy.Gaussian(dim)

problems = {
    "Gauss": {
        "default": (dim, hopsy.add_box_constraints(hopsy.Problem(A, b, gauss), lb, ub)),
        "rounded": (dim, hopsy.round(hopsy.add_box_constraints(hopsy.Problem(A, b, gauss), lb, ub))),
    },
    "STAT-1": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb, ub)) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb, ub))) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
    },
    "STAT-1-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni)) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni))) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
    },
    "STAT-2": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), lb, ub)) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb, ub))) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
    },
    "STAT-2-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni)) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni))) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
    },
    #"INST": {
    #    "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni)) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0], #+
    #    "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), lb_ni, ub_ni))) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0], #+
    #},
}

gauss_lb, gauss_ub = -3, 3
stat_lb, stat_ub = [0, 0], [2, 2]
stat_lb_ni, stat_ub_ni = [0, 0, 0], [2, 2, 100]
inst_lb, inst_ub = 0, 100

small_bound_problems = {
    "Simplicus": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb, stat_ub)) for model in [x3c2.X3CModel("models/Simplicus.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb, stat_ub))) for model in [x3c2.X3CModel("models/Simplicus.fml")]][0],
    },
    "Gauss": {
        "default": (dim, hopsy.add_box_constraints(hopsy.Problem(A, b, gauss), gauss_lb, gauss_ub)),
        "rounded": (dim, hopsy.round(hopsy.add_box_constraints(hopsy.Problem(A, b, gauss), gauss_lb, gauss_ub))),
    },
    "STAT-1": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb, stat_ub)) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb, stat_ub))) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal.fml")]][0],
    },
    "STAT-1-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb_ni, stat_ub_ni)) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb_ni, stat_ub_ni))) for model in [x3c2.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]][0], 
    },
    "STAT-2": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), stat_lb, stat_ub)) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb, stat_ub))) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal.fml")]][0], #+
    },
    "STAT-2-ni": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb_ni, stat_ub_ni)) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), stat_lb_ni, stat_ub_ni))) for model in [x3c2.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]][0], #+
    },
    #"INST": {
    #    "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), inst_lb, inst_ub)) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0], #+
    #    "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model), inst_lb, inst_ub))) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0], #+
    #},
}

starting_points = {
    "Simplicus": {
        "default": [[.3, .2]],
        "rounded": [],
    },
    "Gauss": {
        "default": [[0] * dim],
        "rounded": [],
    },
    "STAT-1": {
        "default": [[.6, 1]],
        "rounded": [],
    },
    "STAT-1-ni": {
        "default": [[.6, 1, .05]],
        "rounded": [], 
    },
    "STAT-2": {
        "default": [[.8, 1], [.2, 1]],
        "rounded": [],
    },
    "STAT-2-ni": {
        "default": [[.7, 1, 0.01], [.1, 1, 0.01]],
        "rounded": [],
    },
    "INST": {
        "default": [[10., 15., 20., 25., 10., 10., 50., 0.6, 1.]],
        "rounded": [],
    },
}

for key in problems:
    T, s = problems[key]['rounded'][1].transformation, problems[key]['rounded'][1].shift
    starting_points[key]['rounded'] = [np.linalg.solve(T, x - s) for x in starting_points[key]['default']]

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

# fine stepsize interval is given as [m-l, m+l]
m, l = 0, 1
a, b = -5, 3

fine, coarse = 16, 5 

def get_stepsize_grid(_m, _l, _a=a, _b=b):
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
    #('STAT-1-ni', 'CSmMALA'): (.5, 1),
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
    "neff": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states, relative=True))]*n_chains),
    "neff/t": (lambda rhat, elapsed, accrate, states: [np.min(hopsy.ess(states, relative=True)) / (elapsed / states.shape[1])]*n_chains),
    "rhat": (lambda rhat, elapsed, accrate, states: [rhat]*n_chains),
    "acc": (lambda rhat, elapsed, accrate, states: np.mean(accrate, axis=0)),
    "esjd": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1)),
    #"esjd": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1)),
    "esjd/t": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1) / (elapsed / states.shape[1])),
    #"esjd/t": (lambda rhat, elapsed, accrate, states: np.mean(np.linalg.norm(np.diff(states, axis=1), axis=-1)**2, axis=-1) / (elapsed / states.shape[1])),
    "t": (lambda rhat, elapsed, accrate, states: [elapsed / states.shape[1]]*n_chains),
    "n": (lambda rhat, elapsed, accrate, states: [states.shape[1]]*n_chains),
}

target_display_names = {
    "neff":   r"min $n_{\mathrm{eff}}$",
    "neff/t": r"min $n_{\mathrm{eff}}/s$",
    "rhat":   r"max $\hat{R}$",
    "acc":    "Acceptance rate",
    "esjd":   "ESJD",
    "esjd/t": "$\mathrm{ESJD}/s$",
    "t":      "$T/n$",
}

opt_sampling = {
    'Gauss': (hopsy.GaussianHitAndRunProposal, 1), 
    'STAT-1': (hopsy.GaussianHitAndRunProposal, 1), 
    'STAT-1-ni': (hopsy.GaussianHitAndRunProposal, 1), 
    'STAT-2': (hopsy.GaussianHitAndRunProposal, 1), 
    'STAT-2-ni': (hopsy.GaussianHitAndRunProposal, 1),
    'INST': (hopsy.AdaptiveMetropolisProposal, 1),
}

n_test_samples = 100

ts_params = {
    "n_posterior_update": 100,
    "lower_bound": 10**a,
    "upper_bound": 10**b,
    "grid_size": (b-a)*10 + 1,
    "record_data": True,
    "n_convergence": 20,
}

tuning_targets = {
    "Acceptance\nRate\n(1-norm)": lambda mcs, dim: hopsy.AcceptanceRateTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order),
    "Acceptance\nRate\n(2-norm)": lambda mcs, dim: hopsy.AcceptanceRateTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order, order=2),
    "ESJD": lambda mcs, dim: hopsy.ExpectedSquaredJumpDistanceTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order, estimate_covariance=False),
    "1,5-ESJD": lambda mcs, dim: hopsy.ExpectedSquaredJumpDistanceTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order, lags=[1, 5], estimate_covariance=False),
    "ESJD/s": lambda mcs, dim: hopsy.ExpectedSquaredJumpDistanceTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order, consider_time_cost=True, estimate_covariance=False),
    "1,5-ESJD/s": lambda mcs, dim: hopsy.ExpectedSquaredJumpDistanceTarget(mcs, n_test_samples=n_test_samples*dim**n_samples_order, consider_time_cost=True, lags=[1, 5], estimate_covariance=False),
}

target_map = {
    "Acceptance\nRate\n(1-norm)": "acc",
    "Acceptance\nRate\n(2-norm)": "acc",
    "ESJD": "esjd",
    "1,5-ESJD": "neff",
    "ESJD/s": "esjd/t",
    "1,5-ESJD/s": "neff/t",
}

tuning_problems = {
    "Spiralus": {
        "default": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), lb, ub)) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0],
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), lb, ub))) for model in [x3c2.X3CModel("models/Spiralus_INST.fml")]][0],
    },
    "Coryne": {
        "default": [(model.A.shape[1], hopsy.Problem(model.A, model.b, model, model.initial_point)) for model in [x3c2.X3CModel("models/Coryne.fml")]][0], 
        "rounded": [(model.A.shape[1], hopsy.round(hopsy.Problem(model.A, model.b, model, model.initial_point))) for model in [x3c2.X3CModel("models/Coryne.fml")]][0], 
    },
}

starting_points["Spiralus"] = {
    "default": [x3c2.X3CModel("models/Spiralus_INST.fml").initial_point],
    "rounded": hopsy.transform(tuning_problems["Spiralus"]["rounded"][1], [x3c2.X3CModel("models/Spiralus_INST.fml").initial_point]),
}

starting_points["Coryne"] = {
    "default": [x3c2.X3CModel("models/Coryne.fml").initial_point],
    "rounded": hopsy.transform(tuning_problems["Coryne"]["rounded"][1], [x3c2.X3CModel("models/Coryne.fml").initial_point]),
}


def uniform_sampling_single_arg(args):
    return uniform_sampling(*args)

def uniform_sampling(problem, dim, starting_points, seed):
    _problem = hopsy.Problem(problem.A, problem.b, transformation=problem.transformation, shift=problem.shift)
    mcs = [hopsy.MarkovChain(_problem, hopsy.UniformHitAndRunProposal, starting_points[i]) for i in range(n_chains)]
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    accrate, states = hopsy.sample(mcs, rngs, int(n_samples), int(100*dim), n_chains)
    rhat = np.max(hopsy.rhat(states))
    
    i = 0
    while not(rhat < rhat_threshold) and i < N_max:
        _accrate, _states = hopsy.sample(mcs, rngs, int(n_samples/100), int(100*dim), n_chains)
        
        states = np.concatenate([states, _states], axis=1)
        rhat = np.max(hopsy.rhat(states))
        
        i += 1
        
        
    thinning = int(states.shape[1] / keep_n_samples)
        
    return states[:,::thinning]

def posterior_sampling_single_arg(args):
    return posterior_sampling(*args)

def posterior_sampling(Proposal, problem, dim, starting_points, stepsize, seed):
    mcs = [hopsy.MarkovChain(problem, Proposal, starting_points[i]) for i in range(n_chains)]
    for mc in mcs: mc.proposal.stepsize = stepsize
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    accrate, states = hopsy.sample(mcs, rngs, n_samples, dim, n_chains)
    rhat = np.max(hopsy.rhat(states))
    
    i = 0
    while not(rhat < rhat_threshold) and i < N_max:
        _accrate, _states = hopsy.sample(mcs, rngs, n_samples, dim, n_chains)
        
        states = np.concatenate([states, _states], axis=1)
        rhat = np.max(hopsy.rhat(states))
        
        i += 1
        
    #thinning = int(states.shape[1] / keep_n_samples)
        
    return states#[:,::thinning]


def bruteforce_sampling(Proposal, problem, dim, starting_points, stepsize, seed):
    """
        Samples problem using a proposal of type Proposal from starting_point and with
        stepsize.
    """
    mcs = [hopsy.MarkovChain(problem, Proposal, starting_points[i]) for i in range(n_chains)]
    for mc in mcs: mc.proposal.stepsize = stepsize
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    elapsed = time.time()
    accrate, states = hopsy.sample(mcs, rngs, dim * n_samples)
    elapsed = time.time() - elapsed
    rhat = np.max(hopsy.rhat(states))
    
    accrate = [accrate]
    
    i = 0
    while not(rhat < rhat_threshold) and i < N_max:
        elapsed = time.time() - elapsed
        _accrate, _states = hopsy.sample(mcs, rngs, dim * n_samples)
        elapsed = time.time() - elapsed
        
        accrate += [_accrate]
        states = np.concatenate([states, _states], axis=1)
        
        rhat = np.max(hopsy.rhat(states))
        
        i += 1
        
    result = []
    for _, target in targets.items():
        result.append(target(rhat, elapsed/n_chains, np.array(accrate), states))
        
    return result


def tuning(Proposal, problem, dim, target, starting_points, seed):
    if starting_points is not None:
        mcs = [hopsy.MarkovChain(problem, Proposal, starting_points[i]) for i in range(n_chains)]
    else:
        mcs = [hopsy.MarkovChain(problem, Proposal) for i in range(n_chains)]
    target_estimator = tuning_targets[target](mcs, dim)
    
    if "Acceptance" in target and Proposal == hopsy.CSmMALAProposal:
        target_estimator.acceptance_rate = 0.574

    ts = hopsy.ThompsonSamplingTuning(**ts_params, random_seed=seed)
    rngs = [hopsy.RandomNumberGenerator(seed, i) for i in range(n_chains)]

    elapsed = time.time()
    stepsize, posterior = hopsy.tune(ts, target_estimator, rngs)
    elapsed = time.time() - elapsed

    return (elapsed, stepsize, (ts.n_converged < ts.n_posterior_updates), posterior)


def get_uniform_args(problems=small_bound_problems):
    args = []
    args_idx = []
    args_key = []

    for problem_key in problems:
        n_jobs = len(args)
        variant = "rounded"
        dim, problem = problems[problem_key][variant]
        
        for seed in range(1):
            rng = hopsy.RandomNumberGenerator(seed, n_chains)
            uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
            draws = [int(uniform(rng)) for i in range(n_chains)]

            _starting_points = [starting_points[problem_key][variant][i] for i in draws]
            
            args += [(problem, dim, _starting_points, seed)]

    args_idx += [problem for problem in problems]

    return args, args_idx


def get_posterior_args(prior=None):
    args = []
    args_idx = []
    args_key = []

    for problem_key in small_bound_problems:
        n_jobs = len(args)
        
        Proposal = opt_sampling[problem_key][0]
        stepsize = opt_sampling[problem_key][1]
        
        variant = "rounded"
        dim, problem = small_bound_problems[problem_key][variant]
        
        for seed in range(1):
            if prior is not None:
                _prior = prior[problem_key].reshape(-1, dim)

                rng = hopsy.RandomNumberGenerator(seed, n_chains)
                uniform = hopsy.Uniform(0, _prior.shape[0])
                draws = [int(uniform(rng)) for i in range(n_chains)]

                _starting_points = [_prior[i] for i in draws]
                T, s = problem.transformation, problem.shift
                _starting_points = [np.linalg.solve(T, x - s) for x in _starting_points]
            else:
                rng = hopsy.RandomNumberGenerator(seed, n_chains)
                uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
                draws = [int(uniform(rng)) for i in range(n_chains)]

                _starting_points = [starting_points[problem_key][variant][i] for i in draws]
            
            args += [(Proposal, problem, dim, _starting_points, stepsize, seed)]

    args_idx += [problem for problem in problems]

    return args, args_idx


def get_bruteforce_args(prior=None):
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
                    if prior is not None:
                        _prior = prior[problem_key].reshape(-1, dim)
                        
                        rng = hopsy.RandomNumberGenerator(seed, n_chains)
                        uniform = hopsy.Uniform(0, _prior.shape[0])
                        draws = [int(uniform(rng)) for i in range(n_chains)]

                        _starting_points = [_prior[i] for i in draws]
                    
                        if variant == "rounded":
                            T, s = problem.transformation, problem.shift
                            _starting_points = [np.linalg.solve(T, x - s) for x in _starting_points]
                    else:
                        rng = hopsy.RandomNumberGenerator(seed, n_chains)
                        uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
                        draws = [int(uniform(rng)) for i in range(n_chains)]

                        _starting_points = [starting_points[problem_key][variant][i] for i in draws]
                        

                    args += [(Proposal, problem, dim, _starting_points, stepsize, seed)]

    args_idx += [(problem, proposal, i, seed) for problem in problems
                                              for proposal in proposals
                                              for i, _ in enumerate(stepsize_grids[(problem, proposal)])
                                              for seed in range(n_seeds)]

    return args, args_idx


def get_tuning_args():
    #def tuning(Proposal, problem, dim, target, starting_points, seed):
    args = []
    args_idx = []

    for problem_key in problems:
        _problems = problems[problem_key]
        for proposal_key, Proposal in proposals.items():
            if rounding[proposal_key]:
                variant = "rounded"
            else:
                variant = "default"

            dim, problem = _problems[variant]

            for target in tuning_targets:
                for seed in range(n_seeds):
                    rng = hopsy.RandomNumberGenerator(seed, n_chains)
                    uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
                    draws = [int(uniform(rng)) for i in range(n_chains)]

                    _starting_points = [starting_points[problem_key][variant][i] for i in draws]
                        
                    args += [(Proposal, problem, dim, target, _starting_points, seed)]
            
    args_idx += [(problem, proposal, target, seed) for problem in problems
                                                   for proposal in proposals
                                                   for target in tuning_targets
                                                   for seed in range(n_seeds)]

    return args, args_idx


def get_pure_tuning_args(prior=None):
    #def tuning(Proposal, problem, dim, target, starting_points, seed):
    args = []
    args_idx = []

    for problem_key in tuning_problems:
        _problems = tuning_problems[problem_key]
        for proposal_key, Proposal in proposals.items():
            if proposal_key == "CSmMALA": continue
            if rounding[proposal_key]:
                variant = "rounded"
            else:
                variant = "default"

            dim, problem = _problems[variant]

            for target in tuning_targets:
                for seed in range(n_seeds):
                    args += [(Proposal, problem, dim, target, None, seed)]
                    args_idx += [(problem_key, proposal_key, target, seed)]

    return args, args_idx


def get_validation_args(prior=None, stepsizes=None):

    args = []
    args_idx = []

    for problem_key in tuning_problems:
        _problems = tuning_problems[problem_key]
        for proposal_key, Proposal in proposals.items():
            if proposal_key == "CSmMALA": continue
            if rounding[proposal_key]:
                variant = "rounded"
            else:
                variant = "default"

            dim, problem = _problems[variant]

            for j, target in enumerate(stepsizes[(problem_key, proposal_key)]):
                for k, stepsize in enumerate(stepsizes[(problem_key, proposal_key)][target][0]):
                    for seed in range(n_seeds):
                        if prior is not None:
                            _prior = prior[problem_key].reshape(-1, dim)

                            rng = hopsy.RandomNumberGenerator(seed, n_chains)
                            uniform = hopsy.Uniform(0, _prior.shape[0])
                            draws = [int(uniform(rng)) for i in range(n_chains)]

                            _starting_points = [_prior[i] for i in draws]

                            if variant == "rounded":
                                T, s = problem.transformation, problem.shift
                                _starting_points = [np.linalg.solve(T, x - s) for x in _starting_points]
                        else:
                            rng = hopsy.RandomNumberGenerator(seed, n_chains)
                            uniform = hopsy.Uniform(0, len(starting_points[problem_key][variant]))
                            draws = [int(uniform(rng)) for i in range(n_chains)]

                            _starting_points = [starting_points[problem_key][variant][i] for i in draws]


                        args += [(Proposal, problem, dim, _starting_points, stepsize[0], seed)]
                        args_idx += [(problem_key, proposal_key, j, k, seed)]

    return args, args_idx
