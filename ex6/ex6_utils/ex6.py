import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, norm
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
import time
from scipy.special import logsumexp
from typing import Tuple



def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float=0, sig_o: float=10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    model.fit(X, y, sample=True)

    k = np.zeros(X.shape)
    k_1 = p_out * norm.pdf(y, loc=mu_o, scale=sig_o)

    for i in range(T):

        k_0 = (1 - p_out) * np.exp(model.log_likelihood(X, y))
        k = k_1 / (k_1 + k_0)

        for j in range(k.shape[0]):
            k[j] = np.random.binomial(1, p=k[j])

        model.fit(X[k == 0], y[k == 0])

    return model, np.array([i for i, k_val in enumerate(k) if k_val])


class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool=True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        self.k = k
        self.alphas = np.full(k, alpha)
        self.mu_0 = mu_0
        self.sig_0 = sig_0
        self.nu = nu
        self.beta = beta
        self.learn_cov = learn_cov
        self.X = None
        self.pi = np.full(k, 1/k)
        self.z = None
        self.mu = np.random.multivariate_normal(mu_0, np.eye(mu_0.shape[0]) * sig_0, size=k)

        self.d = self.mu_0.shape[0]

        if learn_cov:
            self.cov = [self.beta * np.eye(self.d) for _ in range(k)]
        else:  # Fixed covariance
            # for step 3 of gibbs, the second part of the mu[k]
            # distribution mean
            self.mu_numerator_p2 = (1 / self.sig_0) * self.mu_0

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """
        l = np.empty((X.shape[0], self.k))

        if self.learn_cov:
            for k in range(self.k):
                l[:, k] = np.log(multivariate_normal(self.mu[k], self.cov[k]).pdf(X)) \
                          + np.log(self.pi[k])

        else:
            p1 = self.d * np.log(2 * np.pi * self.beta)
            for k in range(self.k):
                p2 = (1 / self.beta) * np.sum((X - self.mu[k]) ** 2, axis=1)
                l[:, k] = -0.5 * (p1 + p2) + np.log(self.pi[k])

        return l

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        return np.argmax(self.log_likelihood(X), axis=1)

    def _update_z(self):
        x_log_like = self.log_likelihood(self.X)

        self.z = np.empty(x_log_like.shape[0])

        for i in range(x_log_like.shape[0]):
            c = x_log_like[i, :]
            c = np.exp(c - logsumexp(c))
            self.z[i] = np.random.choice(range(self.k), 1, p=c)[0]

    def _update_pi(self):
        alphas = self.alphas.copy()
        for val in self.z:
            alphas[int(val)] += 1

        self.pi = np.random.dirichlet(alphas)

    def _update_mu_cov(self):
        if self.learn_cov:
            sig_0_inv = 1 / self.sig_0
            mu_k_mu_p2_2 = sig_0_inv * self.mu_0

            for k in range(self.k):
                k_indices = self.z == k
                cov_k = self.cov[k]

                # calc mu[k] distributions params
                inv_cholesky_cov_k = np.linalg.inv(np.linalg.cholesky(cov_k))
                cov_k_inv = inv_cholesky_cov_k.T @ inv_cholesky_cov_k

                N_k = sum(k_indices)

                # mu[k] current covariance
                mu_k_cov_tmp = N_k * cov_k_inv + sig_0_inv * np.eye(
                    cov_k.shape[0])

                inv_cholesky_mu_k_cov = np.linalg.inv(
                    np.linalg.cholesky(mu_k_cov_tmp))
                mu_k_cov = inv_cholesky_mu_k_cov.T @ inv_cholesky_mu_k_cov

                # mu[k] current mu
                x_k = self.X[k_indices, :]
                mu_k_mu_p2_1 = cov_k_inv @ np.sum(x_k, axis=0)

                mu_k_mu_p2 = mu_k_mu_p2_1 + mu_k_mu_p2_2
                mu_k_mu = np.linalg.solve(mu_k_cov_tmp, mu_k_mu_p2)

                self.mu[k] = np.random.multivariate_normal(mu_k_mu, mu_k_cov)

                # update Cov_k
                x_k_minus_mu_s = x_k - self.mu[k]
                self.cov[k] = (self.nu * self.beta * np.eye(self.d) +
                               x_k_minus_mu_s.T @ x_k_minus_mu_s) / (self.nu
                                                                     + N_k)

        else:  # no learn cov, Fixed covariance
            sig_0_inv = 1 / self.sig_0
            # start = time.time()
            unique_k = np.unique(self.z).astype(int)
            # sample according not exits k values:
            cov_no_k_values = self.sig_0 * np.eye(self.d)
            self.mu = np.random.multivariate_normal(self.mu_0,
                                                  cov_no_k_values,
                                          size=self.k)

            for k in unique_k:
                mu_numerator_p1 = (1 / self.beta) * sum(self.X[self.z == k, :])
                N_k = sum(self.z == k)
                denominator = (N_k / self.beta) + sig_0_inv
                mu = (mu_numerator_p1 + self.mu_numerator_p2) / denominator

                cov = (1 / denominator) * np.eye(mu.shape[0])
                self.mu[k] = np.random.multivariate_normal(mu, cov)


    def gibbs_fit(self, X: np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        self.X = X

        for _ in range(T):

            self._update_z()
            self._update_pi()
            self._update_mu_cov()

        return self



if __name__ == '__main__':
    # ------------------------------------------------------ section 2 - Robust Regression
    # ---------------------- question 2
    # load the outlier data
    x, y = outlier_data(50)
    # init BLR model that will be used to fit the data
    mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)

    # sample using the Gibbs sampling algorithm and plot the results
    plt.figure()
    plt.scatter(x, y, 15, 'k', alpha=.75)
    xx = np.linspace(-0.2, 5.2, 100)
    for t in [0, 1, 5, 10, 25]:
        samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
        plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')

    plt.xlim([np.min(xx), np.max(xx)])
    plt.legend()
    plt.show()

    # ---------------------- question 3
    # load the images to use for classification
    N = 1000
    ims, labs = load_dogs_vs_frogs(N)
    # define BLR model that should be used to fit the data
    mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # use Gibbs sampling to sample model and outliers
    samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # plot the outliers
    plot_ims(ims[outliers], title='outliers')

    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)

    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5,beta=.5)
        gmm.gibbs_fit(X, T=100)

        # plot a histogram of the mixture probabilities (in descending order)
        pi = gmm.pi  # mixture probabilities from the fitted GMM
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.savefig(f"Q5 bar {i + 1}")
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, gmm.mu, np.array(gmm.cov), gmm.cluster(X))
        # the second  input are  the means and the third are the covariances


    # ---------------------- questions 6-7
    # load image data
    # MNIST, labs = load_MNIST()
    # # flatten the images
    # ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    # gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5*np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    # gmm.gibbs_fit(ims, 100)
    #
    # # plot a histogram of the mixture probabilities (in descending order)
    # pi = gmm.pi  # mixture probabilities from the fitted GMM
    # plt.figure()
    # plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    # plt.ylabel(r'$\pi_k$')
    # plt.xlabel('cluster number')
    # plt.show()
    #
    # print("total number of clusters with pi_k > 10^-4: ", sum(pi > 1e-4))
    #
    # # find the clustering of the images to different Gaussians
    # cl = gmm.cluster(ims)
    # clusters = np.unique(cl)
    # print(f'{len(clusters)} clusters used')
    #
    # # calculate the purity of each of the clusters
    # purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    # purity_inds = np.argsort(purities)
    #
    # # plot 25 images from each of the clusters with the top 5 purities
    # for ind in purity_inds[-5:]:
    #     clust = clusters[ind]
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')
    #
    # # plot 25 images from each of the clusters with the bottom 5 purities
    # for ind in purity_inds[:5]:
    #     clust = clusters[ind]
    #     plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')
    #
