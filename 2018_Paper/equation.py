from functools import lru_cache

import numpy as np
import tensorflow as tf
from scipy.linalg import hadamard

from sampler import Brownian, Hadamard


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

        if eqn_config.sampler == "Brownian":
            self.sampler = Brownian(
                self.dim,
                self.num_time_interval,
                self.sqrt_delta_t,
            )
        elif eqn_config.sampler == "Hadamard":
            self.sampler = Hadamard(
                self.dim,
                self.num_time_interval,
                self.sqrt_delta_t,
            )
        else:
            raise NotImplementedError

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""

    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample, x_sample = self.sampler.sample(num_sample, self.x_init)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


class HJBLQ_ORTHO_STEP(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""

    def __init__(self, eqn_config):
        super(HJBLQ_ORTHO_STEP, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def _next_power_of_2(self, n):
        return 1 if n == 0 else 1 << (n - 1).bit_length()  # faster than log2/ceil

    @lru_cache(maxsize=None)
    def _cached_hadamard_power_of_2(self, D):
        return hadamard(D, dtype=np.float32)

    def _get_truncated_hadamard(self, d):
        D = self._next_power_of_2(d)
        return self._cached_hadamard_power_of_2(D)[:d, :]  # only need first d rows

    def sample(self, num_sample):
        # Precompute truncated Hadamard
        H_trunc = self._get_truncated_hadamard(self.dim)
        H_trunc_tf = tf.constant(H_trunc)  # shape: [dim, k]
        k = H_trunc.shape[1]

        # How many increments we need
        total_increments = num_sample * self.num_time_interval
        num_complete_blocks = total_increments // k
        remainder = total_increments % k

        # Efficient permutation generation using argsort of random keys
        if num_complete_blocks > 0:
            rand_keys = tf.random.uniform([num_complete_blocks, k])
            shuffled_blocks = tf.argsort(rand_keys, axis=1)
            cols_complete = tf.reshape(shuffled_blocks, [-1])
        else:
            cols_complete = tf.constant([], dtype=tf.int32)

        if remainder > 0:
            rand_keys_rem = tf.random.uniform([k])
            remainder_cols = tf.argsort(rand_keys_rem)[:remainder]
            cols_tf = tf.concat([cols_complete, remainder_cols], axis=0)
        else:
            cols_tf = cols_complete

        # Gather columns
        dw = tf.gather(H_trunc_tf, cols_tf, axis=1)  # shape: [dim, total_increments]

        # Vectorized Rademacher sampling: {-1, +1}
        rademacher = tf.sign(tf.random.uniform([total_increments], -1, 1, dtype=tf.float32))
        dw = dw * rademacher[tf.newaxis, :] * self.sqrt_delta_t

        # Reshape to [num_sample, dim, num_time_interval]
        dw_sample = tf.reshape(dw, [self.dim, num_sample, self.num_time_interval])
        dw_sample = tf.transpose(dw_sample, [1, 0, 2])

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""

    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample, x_sample = self.sampler.sample(num_sample, self.x_init)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y - tf.pow(y, 3)

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """

    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02  # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample, x_sample = self.sampler.sample(num_sample, self.x_init)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                    self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        piecewise_linear = tf.nn.relu(
            tf.nn.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        return tf.reduce_min(x, 1, keepdims=True)


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar - (self.sigma ** 2) / 2) * self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        temp = tf.reduce_sum(z, 1, keepdims=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
                (self.rb - self.rl) * tf.maximum(temp - y, 0))

    def g_tf(self, t, x):
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 120, 0) - 2 * tf.maximum(temp - 150, 0)


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.dim))
        self.sigma = self.dim + 0.0

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * tf.reduce_sum(z, 1, keepdims=True)

    def g_tf(self, t, x):
        return 1 - 1.0 / (1 + tf.exp(t + tf.reduce_sum(x, 1, keepdims=True) / self.dim))


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        x_square = tf.reduce_sum(tf.square(x), 1, keepdims=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = tf.pow(base, self.alpha)
        derivative = self.alpha * tf.pow(base, self.alpha - 1) * tf.cos(base_alpha)
        term1 = tf.reduce_sum(tf.square(z), 1, keepdims=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
                2.0 * derivative + 4.0 / (self.dim ** 2) * x_square * self.alpha * (
                (self.alpha - 1) * tf.pow(base, self.alpha - 2) * tf.cos(base_alpha) - (
                self.alpha * tf.pow(base, 2 * self.alpha - 2) * tf.sin(base_alpha)
        )
        )
        )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):
        return tf.sin(
            tf.pow(tf.reduce_sum(tf.square(x), 1, keepdims=True) / self.dim, self.alpha))


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """

    def __init__(self, eqn_config):
        super(ReactionDiffusion, self).__init__(eqn_config)
        self._kappa = 0.6
        self.lambd = 1 / np.sqrt(self.dim)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 + self._kappa + np.sin(self.lambd * np.sum(self.x_init)) * np.exp(
            -self.lambd * self.lambd * self.dim * self.total_time / 2)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        exp_term = tf.exp((self.lambd ** 2) * self.dim * (t - self.total_time) / 2)
        sin_term = tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))
        temp = y - self._kappa - 1 - sin_term * exp_term
        return tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.square(temp))

    def g_tf(self, t, x):
        return 1 + self._kappa + tf.sin(self.lambd * tf.reduce_sum(x, 1, keepdims=True))
