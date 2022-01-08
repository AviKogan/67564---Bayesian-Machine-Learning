import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    det_cov_post = np.linalg.det(map_cov)
    det_cov_prior = np.linalg.det(sig)

    p1 = 0.5 * np.log(det_cov_post/det_cov_prior)

    inv_cholesky_prior_cov = np.linalg.inv(np.linalg.cholesky(sig))
    inv_prior_cov = inv_cholesky_prior_cov.T @ inv_cholesky_prior_cov

    p2_1 = (map - mu).T @ inv_prior_cov @ (map - mu)
    H = model.h(X)
    p2_2 = (1/n) * (np.linalg.norm(y - H @ map) ** 2)
    p2_3 = X.shape[0] * np.log(n)

    p2 = -0.5 * (p2_1 + p2_2 + p2_3)
    return p1 + p2 - 2 * np.pi


def main():
    # ------------------------------------------------------ section 2.1
    np.random.seed(100)
    # set up the response functions
    f1 = lambda x: x**2 - 1
    f2 = lambda x: x**3 - 3*x
    f3 = lambda x: x**6 - 15*x**4 + 45*x**2 - 15
    f4 = lambda x: 5*np.exp(3*x) / (1 + np.exp(3*x))
    f5 = lambda x: 2*(np.sin(x*2.5) - np.abs(x))
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        ev_res = []
        models = []

        best_ev = -np.inf
        best_idx = 0
        worst_ev = np.inf
        worst_idx = 0
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            ev_res.append(ev)
            if best_ev < ev:
                best_ev = ev
                best_idx = j

            if worst_ev > ev:
                worst_ev = ev
                worst_idx = j

            models.append(BayesianLinearRegression(mean, cov, noise_var, pbf))

        # plot evidence versus degree and predicted fit
        #  evidence versus degree
        plt.figure()
        plt.plot(degrees, ev_res, 'o')
        plt.xlabel('$degree$')
        plt.ylabel("log evidence")
        plt.title(f"f{i + 1} evidence versus degree")
        plt.savefig(f'f{i + 1} evidence versus degree.png')
        plt.show()
        plt.close()
        # predicted fit

        # plot points and prediction
        plt.figure()
        plt.plot(x, y, 'og', label="True points", markersize=2)

        # worst model
        worst_m = models[worst_idx].fit(x,y)
        pred_worst, std_worst = worst_m.predict(x), worst_m.predict_std(x)
        plt.plot(x, pred_worst, 'm', lw=2, label=f"worst fit, degree"
                                                 f" {degrees[worst_idx]}")
        plt.fill_between(x, pred_worst - std_worst, pred_worst + std_worst,
                         alpha=.5, label="worst confidence")

        # best model
        bst_m = models[best_idx].fit(x,y)
        pred_best, std_best = bst_m.predict(x), bst_m.predict_std(x)
        plt.plot(x, pred_best, 'k', lw=2,
                 label=f"best fit, degree {degrees[best_idx]}")
        plt.fill_between(x, pred_best - std_best, pred_best + std_best,
                         alpha=.5, label="best confidence")


        plt.xlabel('$x$')
        plt.ylabel(fr'$f_{i + 1}(x)$')
        plt.title(f"f{i + 1} best and worst BLR model fit")
        plt.legend()
        plt.savefig(f'f{i + 1} fits.png')
        plt.show()
        plt.close()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    best_ev = -np.inf
    best_ev_noise = 0
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev
        if best_ev < ev:
            best_ev = ev
            best_ev_noise = n

    # plot log-evidence versus amount of sample noise
    plt.figure()
    plt.plot(noise_vars, evs, 'o')
    plt.xlabel(r'$f_{\sigma^2}(x)$')
    plt.ylabel("log evidence")
    plt.title(r"log evidence score as function of $f_{\sigma^2}(x)$"+
              f"\nhighest ev is {round(best_ev, 2)} with noise"
              f" {round(best_ev_noise, 2)}")
    plt.savefig("log evidence score as function of noise")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()



