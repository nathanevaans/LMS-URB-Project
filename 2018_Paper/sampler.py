import numpy as np
import tensorflow as tf
from scipy.linalg import hadamard
from functools import lru_cache


class Brownian:
    def __init__(self, dim, num_time_interval, sqrt_delta_t):
        self.dim = dim
        self.num_time_interval = num_time_interval
        self.sqrt_delta_t = sqrt_delta_t

    def sample(self, num_sample, x_init):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * x_init

        return dw_sample, x_sample


class Hadamard:
    def __init__(self, dim, num_time_interval, sqrt_delta_t):
        self.dim = dim
        self.num_time_interval = num_time_interval
        self.sqrt_delta_t = sqrt_delta_t

    def _next_power_of_2(self, n):
        return 1 if n == 0 else 1 << (n - 1).bit_length()  # faster than log2/ceil

    @lru_cache(maxsize=None)
    def _cached_hadamard_power_of_2(self, D):
        return hadamard(D, dtype=np.float32)

    def _get_truncated_hadamard(self, d):
        D = self._next_power_of_2(d)
        return self._cached_hadamard_power_of_2(D)[:d, :]  # only need first d rows

    def sample(self, num_sample, x_init):
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
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * x_init

        return dw_sample, x_sample
