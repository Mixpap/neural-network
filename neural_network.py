import numpy as np
import numpy.matlib
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
        """
        programmed like this for numerical stability
        :param a:
        :return:
        """
        m = np.max(0,a);#CHECK
        return m + np.log( np.exp(-m) + np.exp(a-m))
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
                 number_of_outputs, lamda,
                 hidden_bias=None,
                 output_bias=None):
        self.input_dim = input_dim
        self.number_of_outputs = number_of_outputs
        self.hidden_layer = Layer(hidden_neurons, hidden_bias)
        self.output_layer = Layer(number_of_outputs, output_bias)
        self.hidden_activation = activation_function(
            hidden_layer_activation_function)
        self.w1_init = np.random.randn(hidden_neurons, 784)
        self.w2_init = np.random.randn(number_of_outputs, hidden_neurons)

    def calculate_cost(self, x, t, lamda, w1, w2, y):
        """
        sum(sum( T.*Y )) - sum(M)  - sum(log(sum(exp(Y - repmat(M, 1, K)), 2)))
          - (0.5*lambda)*sum(sum(W2.*W2));
        :param x:
        :param t:
        :param lamda:
        :param w1:
        :param w2:
        :return:
        """
        #feed forward
        z = self.hidden_activation(x*w1) #check matrix
        #add bias to z
        z_with_bias = np.ones((np.size(z,1),np.size(z,0)+1))
        z_with_bias[:,1:] = z
        # multiplication and transpose
        y = z*w2 #check transpose(normaly w2)
        max_error = np.max(y, 1)
        E = np.sum(np.sum(np.dot(t, y)) - np.sum(max_error)- np.sum(np.log(
            np.sum(np.exp(y-np.matlib.repmat(max_error, 1,
                                             self.number_of_outputs)),
                   2)))-(1/2*lamda)*np.sum(np.sum(w2**2)))
        return E


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



if __name__ == '__main__':
    pass