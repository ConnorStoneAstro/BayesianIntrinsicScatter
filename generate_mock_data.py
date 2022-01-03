import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial


def X(G):
    # The X-axis is some combination of the variables in a "galaxy"
    return (G["var1"] + 2 * G["var2"]) / 4 - G["var4"] + 1


def X_err(G):
    # Compute the classical error propogation for X-axis
    return np.sqrt(
        (G["var1 err"] / 4) ** 2 + (G["var2 err"] / 2) ** 2 + (G["var4 err"]) ** 2
    )


def Y(G):
    # The Y-axis is some combination of the variables in a "galaxy"
    return (5 * G["var2"] + G["var3"]) / 10 + 6


def Y_err(G):
    # Compute the classical error propogation for X-axis
    return np.sqrt((G["var2 err"] / 2) ** 2 + (G["var3 err"] / 10) ** 2)


def sample_phi_params():
    # This is not used in the toy model. But would otherwise be used to provide parameters to "sample_phi"
    return None


def sample_phi(params, G):
    new_G = deepcopy(G)
    # Randomly perturb each variable by its uncertainty
    new_G["var1"] += np.random.normal(loc=0, scale=new_G["var1 err"])
    new_G["var2"] += np.random.normal(loc=0, scale=new_G["var2 err"])
    new_G["var3"] += np.random.normal(loc=0, scale=new_G["var3 err"])
    new_G["var4"] = -new_G["var2"] / 2 + new_G["var3"] / 3
    return new_G


def generate_sample(N=500):
    # three variables sampled from some base distribution
    var1 = np.random.normal(loc=1, scale=4, size=N)
    var2 = np.random.normal(loc=10, scale=5, size=N)
    var3 = np.random.normal(loc=3, scale=1, size=N)
    var4 = -var2 / 2 + var3 / 3
    # The uncertainty for each of the three variables
    var1_err = np.random.uniform(low=0.5, high=1, size=N)
    var2_err = np.random.uniform(low=0.6, high=2, size=N)
    var3_err = np.random.uniform(low=0.5, high=1.5, size=N)
    var4_err = np.sqrt((var2_err / 10) ** 2 + (var3_err / 3) ** 2)

    # Construct the object list where each dictionary in the list is one "galaxy"
    phi_list_true = list(
        {
            "var1": var1[i],
            "var1 err": var1_err[i],
            "var2": var2[i],
            "var2 err": var2_err[i],
            "var3": var3[i],
            "var3 err": var3_err[i],
            "var4": var4[i],
            "var4 err": var4_err[i],
        }
        for i in range(N)
    )

    # Sample the uncertainty distributions to get a set of observations
    phi_list_observed = list(map(partial(sample_phi, None), phi_list_true))

    plt.scatter(
        list(map(X, phi_list_true)),
        list(map(Y, phi_list_true)),
        color="k",
        s=2,
        label="True",
    )
    plt.scatter(
        list(map(X, phi_list_observed)),
        list(map(Y, phi_list_observed)),
        color="r",
        s=1,
        label="Observed",
    )
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    return phi_list_true, phi_list_observed
