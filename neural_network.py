import numpy as np
import random


def activation_function(case, a):
    """

    :param case: The number of the activation function
    :param a: the matrix that needs to be passed through the activation
    function
    :return:
    """
    # h(a) = log(1 + exp^a )
    # h(a) = (exp^a - exp^(-a))/(exp^a + exp^(-a))
    # h(a) = cos(a)
    if case == 1:
        return np.log(1 + np.exp(a), 2)
    elif case == 2:
        return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    elif case == 3:
        return np.cos(a)


class NeuralNetwork:

    def __init__(self, input_dim, hidden_neurons, number_of_outputs,
                 hidden_bias=None,
                 output_bias=None):
        self.input_dim = input_dim
        self.hidden_layer = Layer(hidden_neurons, hidden_bias)
        self.output_layer = Layer(number_of_outputs, output_bias)


    def forward_prop(self, input_data):
        """
        We propagate the input data into the neural network
        :param input_data:
        :return:
        """
        pass


class Layer:

    def __init__(self, num_of_neurons, bias=None):
        self.num_of_neurons = num_of_neurons
        self.neurons = [random.random() for i in range(num_of_neurons)]
        self.bias = bias

    def get_info(self):
        print ', '.join("%s: %s" % item for item in vars(self).items())


class Neuron:

    def __init__(self, weights):
        self.weights = weights

    def get_info(self):
        print ', '.join("%s: %s" % item for item in vars(self).items())
