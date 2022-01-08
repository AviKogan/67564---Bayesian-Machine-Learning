from random import random

import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, GaussianProcess, RBF_kernel, accuracy, Gaussian, plot_ims

def find_MMSE_of_mu(mu_0, sig_0_sq, sig_sq, x: np.ndarray, N):
    nom = (1 / sig_sq) * x.sum(axis=0) + (1 / sig_0_sq) * mu_0
    denom = N * (1 / sig_sq) + (1 / sig_0_sq)
    return nom / denom


def plot_decision_boundry(mu_p, mu_m, x, is_MMSE=False):
    mu_p_minus_mu_m = mu_p - mu_m
    nom_1 = np.linalg.norm(mu_p) ** 2 - np.linalg.norm(mu_m) ** 2

    y = nom_1 / (2 * mu_p_minus_mu_m[1]) - \
                        (mu_p_minus_mu_m[0] / mu_p_minus_mu_m[1]) * x

    # plot the decision boundary
    if is_MMSE:
        plt.plot(x[:0], y[:0], color="black")
        plt.plot(x[:, 1], y[:, 1], color="black", label="MMSE boundary")
    else:
        plt.plot(x, y, color="grey")

def main():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x = np.concatenate([x_p, x_m])

    plt.figure()
    # plot the classes as points
    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label=r"$x^{(+)}$")
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label=r"$x^{(-)}$")
    plt.title(r"Sample of 5 points from $x^{(+)}$ and another 5 from $x^{("
              r"+)}$")
    plt.legend(loc='upper left', fontsize=7)
    plt.savefig("Q1 samples")
    plt.show()


    mu_p_post = find_MMSE_of_mu(mu_p, sig_0, sig, x_p, x_p.shape[0])
    mu_m_post = find_MMSE_of_mu(mu_m, sig_0, sig, x_m, x_m.shape[0])
    # plot the decision boundary
    plt.figure()
    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label=r"$x^{(+)}$")
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label=r"$x^{(-)}$")
    plt.scatter(mu_p[0], mu_p[1], 10, label=r"$\mu_{+}$ prior")
    plt.scatter(mu_m[0], mu_m[1], 10, label=r"$\mu_{-}$ prior")
    plt.scatter(mu_p_post[0], mu_p_post[1], 10, label=r"$\mu_{+}$ posterior")
    plt.scatter(mu_m_post[0], mu_m_post[1], 10, label=r"$\mu_{-}$ posterior")
    plt.title("Decision boundary")
    plot_decision_boundry(mu_p_post, mu_m_post, x, True)
    plt.legend(loc='upper left', fontsize=7)
    plt.savefig("Q1 1 Decision Boundaries")
    plt.show()

    # sample from MMSE and plot 10 decision boundaries

    # plot first the points, mu prior and posterior
    plt.figure()
    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label=r"$x^{(+)}$")
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label=r"$x^{(-)}$")
    plt.scatter(mu_p[0], mu_p[1], 10, label=r"$\mu_{+}$ prior")
    plt.scatter(mu_m[0], mu_m[1], 10, label=r"$\mu_{-}$ prior")
    plt.scatter(mu_p_post[0], mu_p_post[1], 10, label=r"$\mu_{+}$ posterior")
    plt.scatter(mu_m_post[0], mu_m_post[1], 10, label=r"$\mu_{-}$ posterior")
    #calculate both classes posterior covariance
    cov_post = (1 / (x_m.shape[0] * (1 / sig) + 1 / sig_0)) * np.eye(
        x_m.shape[1])

    #sampling mu and plor decision boundaries
    for _ in range(10):
        mu_p_post_i = np.random.multivariate_normal(mu_p_post, cov_post)
        mu_m_post_i = np.random.multivariate_normal(mu_m_post, cov_post)

        plot_decision_boundry(mu_p_post_i, mu_m_post_i, x)

    plot_decision_boundry(mu_p_post, mu_m_post, x, True)
    plt.title("10 Decision Boundaries by sampling the class means from the posteriors")
    plt.legend(loc='upper left', fontsize=7)
    plt.savefig("Q1 10 Decision Boundaries")
    plt.show()


    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()
    plt.figure()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    # create function for fast predictions

    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)

        gauss1, gauss2 = Gaussian(beta, nu).fit(dogs), Gaussian(beta, nu).fit(frogs)

        pred = lambda x: np.clip(gauss1.log_likelihood(x) - gauss2.log_likelihood(x), -25, 25)

        train_score[i] = accuracy(pred(train), labels)
        test_score[i] = accuracy(pred(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.title(r"Accuracy of the QDA classifier as a function of the value of $\nu$")
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.savefig("Q2")
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    kern, sigma = RBF_kernel(.009), .1
    Ns = [250, 500, 1000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ns)), np.zeros(len(Ns))

    gp = None
    for i, N in enumerate(Ns):
        print(f'GP using {N} samples', end='', flush=True)
        # fit a GP regression model to the data
        dogs_train_indices = np.random.choice(dogs.shape[0], N, replace=False)
        frogs_train_indices = dogs_train_indices + dogs.shape[0]
        train_indices = np.concatenate([dogs_train_indices,
                                        frogs_train_indices])

        cur_X_train,  cur_y_train = train[train_indices], labels[train_indices]

        gp = GaussianProcess(kern, sigma).fit(cur_X_train, cur_y_train)

        # calculate accuracies
        train_score[i] = accuracy(gp.predict(train), labels)
        test_score[i] = accuracy(gp.predict(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ns, train_score, lw=2, label='train')
    plt.plot(Ns, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('Q4 accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.savefig("accuracy")
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(gp.predict(dogs_t) / gp.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()







