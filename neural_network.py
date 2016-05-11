import numpy as np
import random


def activation_function(case):
    """
    Returns the corresponding activation function
    :param case: The number of the activation function
    :return: the activation function to be used by the neurons
     in the hidden layer
    """
    # h(a) = log(1 + exp^a)
    # h(a) = (exp^a - exp^(-a))/(exp^a + exp^(-a))
    # h(a) = cos(a)
    def logarithmic(a):
        return np.log(1 + np.exp(a))
    def tanh(a):
        return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    def cosine(a):
        return np.cos(a)
    if case == 1:
        return logarithmic
    elif case == 2:
        return tanh
    elif case == 3:
        return cosine


class NeuralNetwork:

    def __init__(self, input_dim, hidden_layer_activation_function,
                 hidden_neurons,
                 number_of_outputs,
                 hidden_bias=None,
                 output_bias=None):
        self.input_dim = input_dim
        self.hidden_layer = Layer(hidden_neurons, hidden_bias)
        self.output_layer = Layer(number_of_outputs, output_bias)
        self.hidden_activation = activation_function(
            hidden_layer_activation_function)


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
