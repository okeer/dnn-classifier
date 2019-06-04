class BaseOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def compute_update(self, layer, d_params):
        d_weights, d_bias = d_params
        return self.learning_rate * d_weights, self.learning_rate * d_bias
