import numpy as np


class BaseOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def compute_update(self, layer, d_params):
        d_weights, d_bias = d_params
        return self.learning_rate * d_weights, self.learning_rate * d_bias

    def notify_minibatch_started(self):
        pass


class AdamOptimizer(BaseOptimizer):
    EPSILON = 1e-8

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate=learning_rate)

        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2

        self.v = {}
        self.s = {}

    def __init_cache(self, layer, d_params):
        weights, bias = d_params
        weights_shape, bias_shape = weights.shape, bias.shape

        self.v[layer] = {}
        self.v[layer]["d_weights"] = np.zeros(weights_shape)
        self.v[layer]["d_bias"] = np.zeros(bias_shape)

        self.s[layer] = {}
        self.s[layer]["d_weights"] = np.zeros(weights_shape)
        self.s[layer]["d_bias"] = np.zeros(bias_shape)

    def __compute_moving_average_grad(self, value, beta, grad, power_func):
        return beta * value + (1 - beta) * power_func(grad)

    def __compute_bias_corrected(self, value, beta):
        return value / (1 - np.power(beta, self.t))

    def __compute_first_moment_est(self, layer, d_params):
        d_weights, d_bias = d_params

        self.v[layer]["d_weights"] = self.__compute_moving_average_grad(self.v[layer]["d_weights"], self.beta1,
                                                                        d_weights, lambda x: x)
        self.v[layer]["d_bias"] = self.__compute_moving_average_grad(self.v[layer]["d_bias"], self.beta1, d_bias,
                                                                     lambda x: x)

        v_corr_weights = self.__compute_bias_corrected(self.v[layer]["d_weights"], self.beta1)
        v_corr_bias = self.__compute_bias_corrected(self.v[layer]["d_bias"], self.beta1)

        return v_corr_weights, v_corr_bias

    def __compute_second_moment_est(self, layer, d_params):
        d_weights, d_bias = d_params

        self.s[layer]["d_weights"] = self.__compute_moving_average_grad(self.s[layer]["d_weights"], self.beta2,
                                                                        d_weights, lambda x: np.power(x, 2))
        self.s[layer]["d_bias"] = self.__compute_moving_average_grad(self.s[layer]["d_bias"], self.beta2, d_bias,
                                                                     lambda x: np.power(x, 2))

        s_corr_weights = self.__compute_bias_corrected(self.s[layer]["d_weights"], self.beta2)
        s_corr_bias = self.__compute_bias_corrected(self.s[layer]["d_bias"], self.beta2)

        return s_corr_weights, s_corr_bias

    def compute_update(self, layer, d_params):
        if layer not in self.v:
            self.__init_cache(layer, d_params)

        v_corr_weights, v_corr_bias = self.__compute_first_moment_est(layer, d_params)
        s_corr_weights, s_corr_bias = self.__compute_second_moment_est(layer, d_params)

        delta_weights = self.learning_rate * v_corr_weights / np.sqrt(s_corr_weights + AdamOptimizer.EPSILON)
        delta_bias = self.learning_rate * v_corr_bias / np.sqrt(s_corr_bias + AdamOptimizer.EPSILON)

        return delta_weights, delta_bias

    def notify_minibatch_started(self):
        self.t += 1
