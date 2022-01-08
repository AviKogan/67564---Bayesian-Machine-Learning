import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from ex4_utils import BayesianLinearRegression, average_error

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Gibbs': r'Gibbs, $\alpha={}$, $\beta={}$, $\delta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def RFF(beta: float, M: int) -> Callable:
    """
    Function that creates Random Fourier Feature (RFF) basis functions for an RBF kernel
    :param beta: the bandwidth of the kernel that is being approximated
    :param M: the number of random features that will be used to approximate the kernel
    :return: a function that receives as an input data points and returns the basis functions applied to them
    """

    w = np.random.normal(0, beta, M)
    b = np.random.uniform(0, 2 * np.pi, M)

    def h(x: np.ndarray) -> np.ndarray:
        x = x.reshape((x.shape[0], 1))
        res = x * w
        res += b
        return np.cos(res) / np.sqrt(M)
    return h


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        return alpha * np.exp(-beta * np.sum(np.abs(x-y)))
    return kern


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        return alpha * np.exp(-beta * np.sum((x - y) ** 2))
    return kern


def Gibbs_kernel(alpha: float, beta: float, delta: float, gamma: float) -> Callable:
    """
    An implementation of the Gibbs kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        l = lambda u: (RBF_kernel(alpha, beta)(u, delta) + gamma)
        l_x = l(x)
        l_y = l(y)
        l_x_squred_plus_l_y_squred = l_x**2 + l_y**2
        coef = np.sqrt((2 * l_x * l_y)/l_x_squred_plus_l_y_squred)

        return coef * np.exp(-np.sum((x - y) ** 2)/l_x_squred_plus_l_y_squred)

    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        coef = alpha * (2 / np.pi)
        sin_numerator = 2 * beta * (np.dot(x, y) + 1)
        sin_denominator_1 = 1 + 2 * beta * (1 + np.dot(x, x))
        sin_denominator_2 = 1 + 2 * beta * (1 + np.dot(y, y))
        sin_denominator = np.sqrt(sin_denominator_1 * sin_denominator_2)
        return coef * np.arcsin(sin_numerator / sin_denominator)

    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        self.kernel = kernel
        self.noise = noise

        self.mu = None
        self.cov = None

        self.C = None
        self.alpha = None
        self.X_train = None

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.X_train = X

        self.C = self.get_gram(X)
        C_plus_noise = self.C.copy()
        np.fill_diagonal(C_plus_noise, np.diagonal(self.C) + self.noise)

        K_plus_noise_chol_inv = np.linalg.inv(np.linalg.cholesky(C_plus_noise))
        self.cov_inv = K_plus_noise_chol_inv.T @ K_plus_noise_chol_inv

        self.alpha = self.cov_inv @ y

        return self

    def _get_new_X_kernel_matrix(self, X):
        train_N = self.X_train.shape[0]
        test_N = X.shape[0]
        K_X = np.empty((train_N, test_N))

        for i in range(train_N):
            for j in range(test_N):
                K_X[i, j] = self.kernel(self.X_train[i], X[j])

        return K_X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """

        return self.alpha @ self._get_new_X_kernel_matrix(X)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)

        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        K = self._get_new_X_kernel_matrix(X)
        m = K.T @ self.alpha

        C_new = self.get_gram(X)
        cov_new = C_new - K.T @ self.cov_inv @ K
        return np.random.multivariate_normal(m, cov_new)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        C_new = self.get_gram(X)

        K = self._get_new_X_kernel_matrix(X)
        post_cov = C_new - K.T @ self.cov_inv @ K
        return np.sqrt(np.diagonal(post_cov))

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        term1 = -0.5 * np.dot(y, self.alpha)
        term2 = -0.5 * np.log(np.linalg.det(np.eye(X.shape[0]) * self.noise))
        return term1 + term2 - (X.shape[0] / 2) * np.log(2*np.pi)

    def get_gram(self, X):
        N = X.shape[0]
        K = np.zeros((N, N))

        #create upper triangular
        for i in range(0, N):
            for j in range(i+1, N):
                K[i, j] = self.kernel(X[i], X[j])

        # create lower triangular
        K = K + K.T

        # create the diagonal
        for i in range(N):
            K[i, i] = self.kernel(X[i], X[i])
        return K


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([-2.1, -4.3, 0.7, 1.2, 3.9])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        #Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 1, 2],        # insert your
        # parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 2, 1],        # insert your
        # parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 2, 1],                    # insert your
        # parameters, order: alpha, beta
        ['RBF', RBF_kernel, 1, 2],                    # insert your
        # parameters, order: alpha, beta

        # Gibbs kernels
        ['Gibbs', Gibbs_kernel, 1, 0.5, 0, .1],             # insert your
        # parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 1, 2, 0, 0.5],    # insert your
        # parameters, order: alpha, beta, delta, gamma
        ['Gibbs', Gibbs_kernel, 2, 1, 1, 1],    # insert your
        # parameters, order: alpha, beta, delta, gamma

        # # quiz Q5
        # ['Gibbs', Gibbs_kernel, 1, 0.1, 0, 0.1],
        # ['Gibbs', Gibbs_kernel, 50, 0.1, 0, 0.1],
        # ['Gibbs', Gibbs_kernel, 100, 0.1, 0, 0.1],
        # #
        # # # quiz Q6
        # ['Gibbs', Gibbs_kernel, 5, 1, 0, 0.1],  # insert your
        # ['Gibbs', Gibbs_kernel, 5, 10, 0, 0.1],  # insert your
        # ['Gibbs', Gibbs_kernel, 5, 20, 0, 0.1],  # insert your
        #
        # # quiz Q7
        # ['Gibbs', Gibbs_kernel, 5, 0.5, 0, 0.1],  # insert your
        # ['Gibbs', Gibbs_kernel, 5, 0.5, 5, 0.1],  # insert your
        # ['Gibbs', Gibbs_kernel, 5, 0.5, 20, 0.1],  # insert your

        #Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],                   # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 1, 2],                      # insert your
        # parameters, order: alpha, beta
        ['NN', NN_kernel, 2, 1],                      # insert your
        # parameters, order: alpha, beta

        # quiz Q8
        # ['NN', NN_kernel, 1, 1],
        # ['NN', NN_kernel, 5, 1],
        # ['NN', NN_kernel, 10, 1],

        # quiz Q9
        # ['NN', NN_kernel, 0.5, 1],
        # ['NN', NN_kernel, 0.5, 5],
        # ['NN', NN_kernel, 0.5, 10],
        # ['NN', NN_kernel, 0.5, 50],
        # ['NN', NN_kernel, 0.5, 100],
    ]
    noise = 0.05

    # plot all of the chosen parameter settings
    plot_num = 0
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise)

        #plot prior variance and samples from the priors
        plt.figure()

        K = gp.get_gram(xx)
        std = np.sqrt(np.diagonal(K))
        m = np.zeros(xx.shape[0])

        plt.plot(xx, m, label="prior mean")
        plt.fill_between(xx, m - std, m + std, alpha=.5, label="prior CI")

        for _ in range(5):
            f_prior = np.random.multivariate_normal(m, K)
            plt.plot(xx, f_prior)

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title("Prior " +KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        # plt.savefig(f"Prior {p[0]} {plot_num % 3 + 1}")
        plt.show()

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title("Posterior " +KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        # plt.savefig(f"Posterior {p[0]} {plot_num % 3 + 1}")
        plt.show()

        plot_num += 1

    #------------------------------ question 4
    # define range of betas
    betas = np.linspace(.1, 15, 101)
    noise = .15

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.savefig("Q4 1")
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]
    print("min evidence = ", betas[srt[0]])
    print("median evidence = ", betas[srt[(len(evidence)+1)//2]])
    print("max evidence = ", betas[srt[-1]])

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.scatter(x, y, 30, 'k', alpha=.5)

    cur_gp = GaussianProcess(RBF_kernel(1, beta=min_ev), noise).fit(x, y)
    cur_pred = cur_gp.predict(xx)
    plt.plot(xx, cur_pred, lw=2, label='min evidence')
    std = cur_gp.predict_std(xx)
    plt.fill_between(xx, cur_pred - std, cur_pred + std, alpha=.5, label="min CI")

    cur_gp = GaussianProcess(RBF_kernel(1, beta=median_ev), noise).fit(x, y)
    cur_pred = cur_gp.predict(xx)
    plt.plot(xx, cur_pred, lw=2, label='median evidence')
    std = cur_gp.predict_std(xx)
    plt.fill_between(xx, cur_pred - std, cur_pred + std, alpha=.5, label="median CI")

    cur_gp = GaussianProcess(RBF_kernel(1, beta=max_ev), noise).fit(x, y)
    cur_pred = cur_gp.predict(xx)
    plt.plot(xx, cur_pred, lw=2, label='max evidence')
    std = cur_gp.predict_std(xx)
    plt.fill_between(xx, cur_pred - std, cur_pred + std, alpha=.5, label="max CI")

    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.savefig("Q4 2")
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define function and parameters
    f = lambda x: np.sin(x*3)/2 - np.abs(.75*x) + 1
    xx = np.linspace(-3, 3, 100)
    noise = .25
    beta = 2

    # calculate the function values
    np.random.seed(0)
    y = f(xx) + np.sqrt(noise)*np.random.randn(len(xx))

    # ------------------------------ question 5
    # fit a GP model to the data
    gp = GaussianProcess(kernel=RBF_kernel(1, beta=beta), noise=noise).fit(xx, y)

    # calculate posterior mean and confidence interval
    m, s = gp.predict(xx), 2*gp.predict_std(xx)
    print(f'Average squared error of the GP is: {average_error(m, y):.2f}')

    # plot the GP prediction and the data
    plt.figure()
    plt.fill_between(xx, m-s, m+s, alpha=.5)
    plt.plot(xx, m, lw=2)
    plt.scatter(xx, y, 30, 'k', alpha=.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title("GP prediction and the data")
    plt.ylim([-3, 3])
    plt.savefig("Q5")
    plt.show()

    # ------------------------------ question 7
    errors = []
    Ms = [1, 2, 5, 15, 25, 50, 75, 100]
    M_for_plots = [1, 5, 15, 100]
    for M in Ms:
        # fit a BLR model using M random Fourier features then calculate posterior mean and confidence interval
        mdl = BayesianLinearRegression(theta_mean=np.zeros(M), theta_cov=np.eye(M)*M, sig=noise,
                                       basis_functions=RFF(beta, M)).fit(xx, y)
        m, s = mdl.predict(xx), 2*mdl.predict_std(xx)
        errors.append(average_error(m, y))
        print(f'Average squared error with M={M} random features: {errors[-1]:.2f}')

        # plot the fit if M is relevant
        if M in M_for_plots:
            plt.figure()
            plt.fill_between(xx, m-s, m+s, alpha=.5)
            plt.plot(xx, m, lw=2)
            plt.scatter(xx, y, 30, 'k', alpha=.5)
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.ylim([-3, 3])
            plt.title(f'$M={M}$')
            plt.savefig(f"Q7 M_{M}")
            plt.show()

    # plot the errors as a function of M
    plt.figure()
    plt.plot(Ms, errors, lw=2)
    plt.xlabel('$M$')
    plt.ylabel('average squared error')
    plt.savefig("Q7 average squared error")
    plt.show()


if __name__ == '__main__':
    main()



