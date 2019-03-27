import numpy
from scipy.special import gamma
from scipy.special import loggamma as loggamma_im
import time


def loggamma(val):
    return loggamma_im(val).real


class GaussianMixture(object):
    def __init__(self, data, K):
        # K: number of clusters
        self.K = K
        # number of data points
        self.N = len(data)
        # dimensionality of data points
        self.m = len(data[0])

        self.data = data

        # membership indicator matrix
        self.membership = numpy.zeros((self.N, self.K), dtype='uint')
        self.posteriors = numpy.zeros((self.N, self.K), dtype='uint')

        # Init to random clusters.
        for i in range(self.N):
            self.membership[i, numpy.random.randint(self.K)] = 1

        self.alpha = 1000

        self.k_0 = 10.0
        self.mu_0 = numpy.ones(self.m) / 10
        self.lambda_0 = numpy.ones((self.m, self.m)) / 10
        self.v_0 = 0.1

    def fit(self, iterations=300, burn_in=30):
        for i in xrange(burn_in):
            self.gibbs_round()

        for i in xrange(iterations):
            self.gibbs_round()
            self.posteriors += self.membership

    def gibbs_round(self):
        for i in xrange(self.N):
            self.gibbs_step_log(i)

    def gibbs_step_log(self, index):
        log_probs = []
        for k in xrange(self.K):
            log_p = self.calculate_log_posterior(index, k)
            log_probs.append(log_p)

        max_log = numpy.max(log_probs)
        probs = [numpy.exp(x - max_log) for x in log_probs]
        probs_sum = numpy.sum(probs)
        norm_probs = [x / probs_sum for x in probs]

        dest_cluster = numpy.random.choice(range(self.K), p=norm_probs)
        source_cluster = numpy.argmax(self.membership[index, :])
        if source_cluster != dest_cluster:
            self.membership[index, source_cluster] = 0
            self.membership[index, dest_cluster] = 1

    def gibbs_step(self, index):
        probs = []
        for k in xrange(self.K):
            prob = self.calculate_posterior(index, k)
            probs.append(prob)

        prob_sum = numpy.sum(probs)
        probs = [x / prob_sum for x in probs]

        dest_cluster = numpy.random.choice(range(self.K), p=probs)
        self.update_membership(index, dest_cluster)

    def update_membership(self, index, dest_cluster):
        # update S
        self.membership[index, :] = 0
        self.membership[index, dest_cluster] = 1

    def prior(self, i, k):
        c_k = numpy.sum(self.membership[:, k])
        if self.membership[i, k] == 1:
            c_k -= 1

        alpha = float(self.alpha)
        K = float(self.K)
        c_k = float(c_k)
        N = float(self.N)

        prior = ((alpha / K) + c_k) / (alpha + N - 1)
        return prior

    def likelihood(self, i, k):
        x = self.data[i]

        c = self.c(i, k)  # cluster specific parameter
        a = self.mu(i, k)  # cluster specific parameter
        B = self.B(i, k)  # cluster specific parameter
        m = self.m  # dimensionality?

        likelihood = \
            gamma((c + m) / 2) / (gamma(c / 2) * (c * numpy.pi) ** (float(m) / 2)) * \
            1.0 / numpy.sqrt(numpy.linalg.det(B)) * \
            (1 + numpy.dot(numpy.dot(x - a, numpy.linalg.inv(B)), x - a) / c) ** (float(- c - m) / 2)

        return likelihood

    def calculate_posterior(self, i, k):
        """
        # Calculate the posterior probability that data point at index i
        # belongs to cluster k
        """
        prior = self.prior(i, k)
        likelihood = self.likelihood(i, k)
        posterior = prior * likelihood

        return posterior

    def calculate_log_posterior(self, i, k):
        x = self.data[i]
        c_k = numpy.sum(self.membership[:, k])
        if self.membership[i, k] == 1:
            c_k -= 1

        log_prior_denominator = numpy.log(self.alpha + self.N - 1)
        log_prior = numpy.log((self.alpha / self.K) + c_k) - log_prior_denominator

        c = self.c(i, k)  # cluster specific parameter
        a = self.mu(i, k)  # cluster specific parameter
        B = self.B(i, k)  # cluster specific parameter
        m = self.m  # dimensionality?

        log_likelihood = \
            loggamma((c + m) / 2) - loggamma(c / 2) - m * numpy.log(c * numpy.pi) / 2 - \
            numpy.log(numpy.sqrt(numpy.linalg.det(B))) + \
            numpy.log((1 + numpy.dot(numpy.dot(x - a, numpy.linalg.inv(B)), x - a) / c) ** (float(- c - m) / 2))

        log_posterior = log_prior + log_likelihood

        return log_posterior

    def c(self, i, k):
        return self.v_k(i, k) - self.m + 1

    def v_k(self, i, k):
        v_0 = self.v_0
        c_k = self.c_k(i, k)

        return v_0 + c_k

    def k_k(self, i, k):
        k_0 = self.k_0
        c_k = self.c_k(i, k)
        return k_0 + c_k

    def mu(self, i, k):
        k_0 = self.k_0
        mu_0 = self.mu_0

        c_k = self.c_k(i, k)
        x_k = self.x_k(i, k)

        assert len(x_k) == self.m

        return (k_0 * mu_0 + c_k * x_k) / (k_0 + c_k)

    def S(self, i, k):
        S = numpy.zeros((self.m, self.m))
        x_k = self.x_k(i, k)
        for idx in numpy.where(self.membership[:, k])[0]:
            if idx != i:
                x = self.data[idx]
                delta_x = x - x_k
                S += numpy.outer(delta_x.T, delta_x)

        return S

    def c_k(self, i, k):
        c_k_vec = self.membership[:, k].copy()
        c_k_vec[i] = 0
        return numpy.sum(c_k_vec)

    def x_k(self, i, k):
        c_k_vec = self.membership[:, k].copy()
        c_k_vec[i] = 0
        x_k = numpy.mean(self.data[c_k_vec == 1], axis=0)
        return x_k

    def lambda_(self, i, k):
        lambda_0 = self.lambda_0
        S = self.S(i, k)
        k_0 = self.k_0
        c_k = self.c_k(i, k)
        x_k = self.x_k(i, k)
        k_k = self.k_k(i, k)
        mu_0 = self.mu_0

        outer = numpy.outer(x_k - mu_0, x_k - mu_0)
        outer_product_term = k_0 * c_k / k_k * outer
        lambda_k = lambda_0 + S + outer_product_term

        return lambda_k

    def B(self, i, k):
        lambda_k = self.lambda_(i, k)
        k_k = self.k_k(i, k)
        v_k = self.v_k(i, k)
        m = self.m

        B = lambda_k * (1 + k_k) / (k_k * (v_k - m + 1))
        return B
