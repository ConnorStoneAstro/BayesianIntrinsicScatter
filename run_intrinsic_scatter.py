from generate_mock_data import (
    X,
    X_err,
    Y,
    Y_err,
    sample_phi_params,
    sample_phi,
    generate_sample,
)
from Stoneetal2021 import BayesianIntrinsicScatter
from BayesInterval import bayes_interval
from scipy import odr
import numpy as np
import matplotlib.pyplot as plt


def relation_f(params, x, y):
    # compute residual between linear fit and observed y-axis value
    return y - (params[0] + x * params[1])


def relation_f_fit(xx, yy):
    # Fit the data with an orthogonal distance regression
    mydata = odr.Data(xx, yy)
    myodr = odr.ODR(mydata, odr.polynomial(1), beta0=[1.0, 1.0])
    res_m = myodr.run()
    return res_m.beta


if __name__ == "__main__":

    # Randomly generate data sample of "galaxies"
    true, observed = generate_sample()

    # Fit observed data and determine total scatter
    params = relation_f_fit(list(map(X, observed)), list(map(Y, observed)))
    residuals = list(relation_f(params, X(O), Y(O)) for O in observed)
    total_scatter = np.std(residuals)

    # Perform Bayesian Intrinsic Scatter analysis
    S, P = BayesianIntrinsicScatter(
        observed,
        total_scatter,
        X,
        Y,
        sample_phi_params,
        sample_phi,
        relation_f,
        relation_f_fit,
    )

    # Perform classical intrinsic scatter analysis
    classical_sigma2avg = np.sqrt(
        np.mean(
            (params[1] * np.array(list(map(X_err, observed)))) ** 2
            + np.array(list(map(Y_err, observed))) ** 2
        )
    )
    classical_int_scatter = np.sign(
        total_scatter ** 2 - classical_sigma2avg ** 2
    ) * np.sqrt(np.abs(total_scatter ** 2 - classical_sigma2avg ** 2))

    # Fit true data and compute intrinsic scatter
    params = relation_f_fit(list(map(X, true)), list(map(Y, true)))
    residuals = list(relation_f(params, X(O), Y(O)) for O in true)
    intrinsic_scatter = np.std(residuals)

    # Plot posterior and intrinsic scatter
    region = bayes_interval(S,P)
    plt.plot(S, P, color="r", linewidth=2, label="Posterior PDF")
    for i,r in enumerate(region):
        plt.axvline(
            r,
            color="r",
            linewidth=1,
            linestyle = '--',
            label="Credible Region" if i == 0 else None,
        )    
    plt.axvline(
        intrinsic_scatter,
        color="k",
        linewidth=2,
        label="true $\\sigma_i = %.2f$" % intrinsic_scatter,
    )
    plt.axvline(
        max(0, classical_int_scatter),
        color="g",
        linewidth=2,
        linestyle="--",
        label="classical $\\sigma_i = %.2f$" % classical_int_scatter,
    )
    plt.legend()
    plt.xlabel("Intrinsic Scatter [$\\sigma_i$]")
    plt.ylabel("Probability Density")
    plt.show()
