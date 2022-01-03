import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm
from functools import partial
from multiprocessing import Pool


def _pdf(r, s):
    return np.sum(norm.pdf(r, loc=0, scale=s))


def BayesianIntrinsicScatter(
    phi_list,
    sigma_max,
    X,
    Y,
    sample_phi_params,
    sample_phi,
    relation_f,
    relation_f_fit,
    N_samples=500,
    N_sigma=100,
    sigma_min=0.01,
    nprocs=4,
    min_pass=100,
):

    pool = Pool(nprocs)
    # array of values at which to evaluate the intrinsic scatter pdf
    S = np.linspace(sigma_min, sigma_max, N_sigma)
    residuals = [[] for n in range(len(phi_list))]

    for r in range(N_samples):
        params = sample_phi_params()
        # Resample galaxy list
        sample = pool.map(partial(sample_phi, params), phi_list)
        # Evaluate scaling relation axes
        XY = zip(pool.map(X, sample), pool.map(Y, sample), range(len(phi_list)))
        XY = list(filter(lambda xy: not None in xy, XY))
        # Fit scaling relation
        fit = relation_f_fit(list(xy[0] for xy in XY), list(xy[1] for xy in XY))

        for xy in XY:
            # Store scaling relation residual
            residuals[xy[2]].append(relation_f(fit, xy[0], xy[1]))
    posteriors = []

    for r in filter(lambda r: len(r) > min_pass, residuals):
        # evaluate the intrinsic scatter pdf for this galaxy
        pdf = np.array(pool.map(partial(_pdf, r), S))
        # normalize pdf to integral 1
        posteriors.append(np.log10(pdf / trapz(pdf, S)))

    # Take product of single galaxy posteriors and normalize
    P_sigmai = np.sum(posteriors, axis=0)
    P_sigmai -= np.max(P_sigmai)
    P_sigmai = (10 ** P_sigmai) / trapz(10 ** P_sigmai, S)

    return S, P_sigmai
