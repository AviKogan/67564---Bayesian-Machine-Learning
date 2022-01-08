import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        return np.concatenate([x[:, None] ** i for i in range(degree + 1)], axis=1)
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        N = x.shape[0]
        X = np.ones(N).reshape(-1, 1)
        for i in range(centers.shape[0]):
            cur = np.exp(-1 * ((x - centers[i]) ** 2) / (2 * (beta ** 2)))
            cur = cur.reshape(-1, 1)
            X = np.concatenate((X, cur), axis=1)
        return X
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        X = np.concatenate([x[:, None] ** i for i in range(4)], axis=1)
        for i in range(knots.shape[0]):
            cur = (np.power(x - knots[i], 3))
            cur = cur.reshape(-1, 1)
            cur[cur < 0] = 0
            X = np.concatenate((X, cur), axis=1)
        return X
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.theta)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.prior_mu = theta_mean
        self.prior_cov = theta_cov
        self.sigma = sig
        self.basis_funcs = basis_functions
        self.posterior_mu = None
        self.posterior_cov = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_funcs(X)
        HT_H = H.T @ H

        inv_cholesky_prior_cov = np.linalg.inv(np.linalg.cholesky(self.prior_cov))
        inv_prior_cov = inv_cholesky_prior_cov.T @ inv_cholesky_prior_cov

        sig_square_inv = 1 / self.sigma

        cholesky_c_inv = np.linalg.inv(np.linalg.cholesky(sig_square_inv * HT_H+ inv_prior_cov))
        self.posterior_cov = cholesky_c_inv.T @ cholesky_c_inv

        self.posterior_mu = self.posterior_cov @ (sig_square_inv * H.T @ y +
                                                  inv_prior_cov @ self.prior_mu)
        # inv_prior_cov = np.linalg.inv(self.prior_cov)
        # self.posterior_cov = np.linalg.inv((1 / self.sigma) * H.T @ H + inv_prior_cov)
        # self.posterior_mu = self.posterior_cov @ ((1 / self.sigma) * H.T @ y
        #                                           + inv_prior_cov @ self.prior_mu)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_funcs(X)@self.posterior_mu

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_funcs(X)
        return np.sqrt(np.diagonal(H@self.posterior_cov@H.T) + self.sigma)

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_funcs(X)
        return H @ np.random.multivariate_normal(self.posterior_mu, self.posterior_cov)


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_funcs = basis_functions
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_funcs(X)
        self.theta = np.linalg.pinv(H) @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_funcs(X) @ self.theta

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_regression_line(x_test, y_test, trained_model, title):
    # plotting the actual points as scatter plot
    plt.scatter(x_test, y_test, color="m", marker="o", label='True points')

    x_line = np.linspace(x_test.min(), x_test.max(), x_test.shape[0])
    y_pred = trained_model.predict(x_test)
    # Plotting the regression line
    plt.plot(x_line, y_pred, color="g", label='Fitted line')
    plt.scatter(x_test, y_pred, color="y", marker="o", label='Predicted points')
    plt.legend(loc="upper left")
    # Putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

def plot_prior(x, mu, cov, sigma, basis_func, title):
        plt.figure()
        H = basis_func(x)
        mean = H@mu
        std = np.sqrt(np.diagonal(H @ cov @ H.T) + sigma)

        plt.fill_between(x, mean - std, mean + std, alpha=.5,
                         label='confidence interval')
        plt.plot(x, mean, label='mean function')

        for _ in range(5):
            mu_samp = np.random.multivariate_normal(mu, cov)
            plt.plot(x, H@mu_samp)
        plt.title(title)
        plt.xlabel(r'$Hour$')
        plt.ylabel(r'$Temperature$')
        plt.legend()
        plt.show()


def plot_posterior(blr, basis_func, test_hours, test, title):
    plt.figure()

    mean = basis_func(test_hours) @ blr.posterior_mu
    std = blr.predict_std(test_hours)
    plt.fill_between(test_hours, mean - std, mean + std, alpha=.5,
                     label='confidence interval')

    plt.plot(test_hours, mean, label='MMSE prediction')
    plt.scatter(test_hours, test, label='True')
    for _ in range(5):
        samp_pred = blr.posterior_sample(test_hours)
        plt.plot(test_hours, samp_pred)

    plt.title(title)
    plt.xlabel(r'$Hour$')
    plt.ylabel(r'$Temperature$')
    plt.legend()
    plt.show()

def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]
    np.random.seed(42)
    #----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with Classical LR with polynomial basic '
              f'functions '
              f'is {np.mean((test - ln.predict(test_hours))**2):.2f}')

        # plot graphs for linear regression part
        plt_title = f"Classical LR with polynomial basic functions {d}"
        plot_regression_line(test_hours, test, ln, plt_title)

    # ----------------------------------------- Bayesian Linear Regression
    # load the historic data

    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 3  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([3, 6, 9, 12, 15, 18, 21])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        #plot prior graphs #######
        title = f"Mean function described by the prior,\npolynomial basic " \
                f"functions of degree {deg}"
        plot_prior(x, mu, cov, sigma, pbf, title)

        # plot posterior graphs ######
        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        blr.fit(train_hours, train)

        title = f"Mean function described by the posterior,\npolynomial " \
                f"basic functions of degree {deg}"
        plot_posterior(blr, pbf, test_hours, test, title)


    # print average squared error performance
        print(f'Average squared error with BLR with polynomial basic functions of degree {deg}'
              f'is {np.mean((test - blr.predict(test_hours))**2):.2f}')


    # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        # plot prior graphs #######
        title = f"Mean function described by the prior,\ngaussian basic " \
                f"function with {c.shape[0]} centers"
        plot_prior(x, mu, cov, sigma, rbf, title)

        #     # plot posterior graphs ######
        blr = BayesianLinearRegression(mu, cov, sigma, rbf)
        blr.fit(train_hours, train)

        title = f"Mean function described by the posterior,\ngaussian basic " \
                f"function with {c.shape[0]} centers"
        plot_posterior(blr, rbf, test_hours, test, title)

        # print average squared error performance
        print(f'Average squared error of BlR with gaussian basic '
              f'functions'
              f' with {c.shape[0]} centers is'
              f' {np.mean((test - blr.predict(test_hours)) ** 2):.2f}')

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        # plot prior graphs #######
        title = f"Mean function described by the prior,\ncubic regression " \
                f"basic functions with {k.shape[0]} knots"
        plot_prior(x, mu, cov, sigma, spline, title)

        #     # plot posterior graphs ######
        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        blr.fit(train_hours, train)

        title = f"Mean function described by the posterior,\ncubic regression " \
                f"basic functions with {k.shape[0]} knots"
        plot_posterior(blr, spline, test_hours, test, title)

        print(f'Average squared error of BlR with cubic regression basic '
              f'functions '
              f'with {k.shape[0]} knots'
              f' {np.mean((test - blr.predict(test_hours)) ** 2):.2f}')


if __name__ == '__main__':
    main()
