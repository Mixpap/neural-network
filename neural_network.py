import numpy as np
import numpy.matlib
import random

def softmax(w):
    return np.exp(w) / np.sum(np.exp(w))


def activation_function(case):
    """
    Returns the corresponding activation function and the derrivative of
    this function
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
        return m + np.log( np.exp(-m) + np.exp(a-m)), 1/1+np.exp(-a)
    def tanh(a):
        return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a)), 1- (np.exp(
            a)-np.exp(-a))/(np.exp(a)+np.exp(-a))**2
    def cosine(a):
        return np.cos(a), -np.sin(a)
    if case == 1:
        return logarithmic
    elif case == 2:
        return tanh
    elif case == 3:
        return cosine


class NeuralNetwork:

    def __init__(self, x_train, hidden_layer_activation_function,
                 hidden_neurons,
                 number_of_outputs, lamda, iter, t, eta, tol,
                 hidden_bias=None,
                 output_bias=None):
        self.iter = iter
        self.x_train = x_train
        self.t = t
        self.lamda = lamda
        self.eta = eta
        self.tol = tol
        self.number_of_outputs = number_of_outputs
        self.hidden_activation, self.grad_activation = activation_function(
            hidden_layer_activation_function)
        #initialize weights
        self.w1 = np.random.randn(hidden_neurons, np.size(input,0)+1)
        self.w2 = np.random.randn(number_of_outputs, hidden_neurons+1)

    def forward_prop(self, x, t, w1, w2):
        """
        feed forward and get error
        sum(sum( T.*Y )) - sum(M)  - sum(log(sum(exp(Y - repmat(M, 1, K)), 2)))
          - (0.5*lambda)*sum(sum(W2.*W2));
        :param x:
        :param t:
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
                   2)))-(1/2*self.lamda)*np.sum(np.sum(w2**2)))
        s = softmax(y)
        gradw2 = np.transpose((t-s))*z_with_bias - self.lamda*w2
        #get rid of the bias
        w2 = w2[:, 2:]
        gradw1 = w2*np.transpose(t-s)*self.grad_activation(x*w1)*x
        return E, gradw1, gradw2

    def train(self):
        e_old = -np.inf
        for i in range(iter):
            e, gradw1, gradw2 = self.forward_prop(self.x_train, self.t,
                                                  self.w1,
                                             self.w2)
            print i," iteration cost =",e
            if abs(e - e_old) < self.tol:
                break
            self.w1 = self.w1 + self.eta*gradw1
            self.w2 = self.w2 + self.eta*gradw2
            e_old = e



"""
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

"""

if __name__ == '__main__':
    train_files = ['data/train%d.txt'% (i,) for i in range(10)]
    test_files = ['data/test%d.txt'% (i,) for i in range(10) ]
    counter = 0
    for i in test_files:
        with open(i, 'r') as fp:
            counter += len(fp.readlines())
    print "train_data = ", counter
    b = np.array(zeros[0].split(" ")).reshape(28,28)
    img = Image.fromarray(b, 'RGB')
    imgplot = plt.imshow(img)
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    print len(tmp)
    train_data = np.array([[j for j in i.split(" ")] for i in tmp])
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    print len(tmp)
    test_data = np.array([[j for j in i.split(" ")] for i in tmp])