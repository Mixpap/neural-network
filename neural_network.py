from __future__ import division
import numpy as np
import numpy.matlib


def softmax(w):
    max_error = np.max(w, 1)
    m = np.array([max_error, ]*w.shape[1]).T
    w = w - m
    w = np.exp(w)
    return w/(np.array([np.sum(w, 1),]*w.shape[1]).T)


def load_data():
    train_files = ['data/train%d.txt' % (i,) for i in range(10)]
    test_files = ['data/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load train data in N*D array (60000x784 for MNIST)
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int')
    print "Train data array size: ", train_data.shape
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST)
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int')
    print "Test data array size: ", test_data.shape
    tmp = []
    for i, _file in enumerate(train_files):
        # print i, _file
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
            # tried this way but didnt work
            # truth.append([1 if j == i else 0 for j in range(0,10)] for
                # line in fp)
    train_truth = np.array(tmp, dtype='int')
    del tmp[:]
    for i, _file in enumerate(test_files):
        # print i, _file
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
            # tried this way but didnt work
            # truth.append([1 if j == i else 0 for j in range(0,10)] for
                # line in fp)
    test_truth = np.array(tmp, dtype='int')
    print "Train truth array size: ", train_truth.shape
    print "Test truth array size: ", test_truth.shape
    return train_data, test_data, train_truth, test_truth


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
        if np.isinf(a).any() : print "we have an inf"; raise RuntimeError
        if np.isnan(a).any() : print "we have an nan"; raise RuntimeError
        m = np.maximum(0, a)  #CHECK
        #print "m is" , m
        return m + np.log(np.exp(-m) + np.exp(a-m))

    def grad_logarithm(a):
        """max_error = np.max(a, 1)
        m = np.array([max_error, ]*a.shape[1]).T
        print "a is:", a.shape, "m is: ", m.shape
        a = a - m
        a = np.exp(a)
        tmp = np.exp(-m)
        return a/(tmp+a)
        return a/(np.array([tmp+a,]*a.shape[1]).T)"""
        return 1/(1+np.exp(-a))

    def tanh(a): #  TODO check stability
        # return (np.exp(a)-np.exp(-a)) / (np.exp(a)+np.exp(-a))
        return (np.exp(2*a)+1)/(np.exp(2*a)-1)

    def grad_tanh(a):
        return 1 - (np.exp(a)-np.exp(-a)) / (np.exp(a)+np.exp(-a))**2

    def cosine(a):
        return np.cos(a)

    def grad_cosine(a):
        return -np.sin(a)

    if case == 1:
        return logarithmic, grad_logarithm
    elif case == 2:
        return tanh, grad_tanh
    elif case == 3:
        return cosine, grad_cosine


class NeuralNetwork:

    def __init__(self, x_train, hidden_layer_activation_function,
                 hidden_neurons, lamda, iter, t, eta, tol,
                 hidden_bias=None,
                 output_bias=None):
        self.iter = iter
        x_with_bias = np.ones((np.size(x_train, 0), np.size(x_train, 1)+1))
        x_with_bias[:, 1:] = x
        self.x = x_with_bias
        self.t = t
        self.lamda = lamda
        self.eta = eta
        self.tol = tol
        self.hidden_neurons = hidden_neurons
        self.number_of_outputs = t.shape[1]
        self.hidden_activation, self.grad_activation = activation_function(
            hidden_layer_activation_function)
        # initialize weights
        self.w1 = np.random.randn(self.hidden_neurons, np.size(self.x, 1))
        print "W(1) is of size M x(D+1) :", self.w1.shape
        # print type(self.w1), type(self.w1[0]), type(self.w1[0,0])
        self.w2 = np.random.randn(self.number_of_outputs, self.hidden_neurons+1)
        print "W(2) is of size K x(M+1) :", self.w2.shape

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
        # feed forward
        x = np.ones((np.size(x, 0), np.size(x, 1)))
        if np.isinf(np.dot(x, w1.T)).any(): print "we have an inf"; raise RuntimeError
        if np.isnan(np.dot(x, w1.T)).any(): print "we have an nan"; raise RuntimeError
        z = self.hidden_activation(np.dot(x, w1.T))
        # check matrix
        # add bias to z
        z_with_bias = np.ones((np.size(z, 0), np.size(z, 1)+1))
        z_with_bias[:, 1:] = z
        # multiplication and transpose
        y = np.dot(z_with_bias, w2.T)  # check transpose(normaly w2)
        max_error = np.max(y, 1)
        E = np.sum(t*y) - np.sum(max_error) - \
            np.sum(
            np.log(
            np.sum(np.exp(y - np.array([max_error, ]*self.number_of_outputs).T),
                   1))) - (0.5*self.lamda)*np.sum(w2*w2)
        """ alternative way to compute E with scipy
        E = np.sum(t*y) - np.sum(max_error) - np.sum(scipy.misc.logsumexp(y - np.array([max_error,
                                                                                        ]*self.number_of_outputs).T)) - (0.5*self.lamda)*np.sum(np.sum(w2*w2))
        """
        s = softmax(y)
        gradw2 = np.dot((t-s).T, z_with_bias) - self.lamda*w2
        # get rid of the bias
        # first =w2[:, 1:].T.dot((t-s).T)
        # second = first*self.grad_activation(x_with_bias.dot(w1.T)).T
        # final = second.dot(x_with_bias)

        if np.isinf(self.grad_activation(x.dot(w1.T))).any() : print "we have an inf"; raise RuntimeError
        if np.isnan(self.grad_activation(x.dot(w1.T))).any() : print "we have an nan"; raise RuntimeError
        gradw1 = (w2[:, 1:].T.dot((t-s).T)*self.grad_activation(x.dot(w1.T)).T).dot(x)
        # print "gradw1 shape: ", gradw1.shape, "||| gradw1 shape:", gradw2.shape
        return E, gradw1, gradw2

    def train(self):
        e_old = -np.inf
        for i in range(iter):
            error, gradw1, gradw2 = self.forward_prop(self.x, self.t, self.w1, self.w2)
            print "iteration #", i, ",error =", error, ", gradw1 : ", gradw1[0, 0], ", gradw2 :", gradw2[0, 0]
            if np.absolute(error - e_old) < self.tol:
                break
            self.w1 += self.eta*gradw1
            self.w2 += self.eta*gradw2
            e_old = error

    def test(self, t_test):
        x = self.x
        # feed forward
        z = self.hidden_activation(np.dot(x, self.w1.T))
        # check matrix
        # add bias to z
        z_with_bias = np.ones((np.size(z, 0), np.size(z, 1)+1))
        z_with_bias[:, 1:] = z
        # multiplication and transpose
        y = np.dot(z_with_bias, self.w2.T)
        s = softmax(y)
        # print s
        decision = np.argmax(s, 1)
        error_count = 0
        for i in range(len(t_test)):
            if np.argmax(t_test[i]) != decision[i]:
                error_count += 1
        print "errors: ", error_count
        print "Error is ", error_count/t_test.shape[0]*100, " %"

    def gradcheck(self):

        epsilon = np.finfo(float).eps
        _list =np.random.randint(self.x.shape[0], size=20)
        x_sample = np.array(self.x[_list, :])
        t_sample = np.array(self.t[_list, :])
        #x_sample = np.array([self.x[i] for i in tmp_list])
        #t_sample = np.array([self.t[i] for i in tmp_list])
        E, gradw1, gradw2 = self.forward_prop(x_sample, t_sample , self.w1, self.w2)  # TODO lamda should be here
        print "gradw2 : ", gradw2.shape
        print "gradw1 : ", gradw1.shape
        numerical_grad_1 = np.zeros(gradw1.shape)
        numerical_grad_2 = np.zeros(gradw2.shape)

        # gradcheck for w1
        for k in range(0, numerical_grad_1.shape[0]):
            for d in range(0, numerical_grad_1.shape[1]):
                w_tmp = np.copy(self.w1)
                w_tmp[k, d] = w_tmp[k, d] + epsilon
                e_plus, _, _ = self.forward_prop(x_sample, t_sample, w_tmp, self.w2)

                w_tmp = np.copy(self.w1)
                w_tmp[k, d] = w_tmp[k, d] - epsilon
                e_minus, _, _ = self.forward_prop(x_sample, t_sample, w_tmp, self.w2)
                numerical_grad_1[k, d] = (e_plus - e_minus) / (2 * epsilon)

                # Absolute norm
        print "The absolute norm for w1 is : ", (np.amax(np.abs(gradw1 - numerical_grad_1)),)

        # gradcheck for w2
        for k in range(0, numerical_grad_2.shape[0]):
            for d in range(0, numerical_grad_2.shape[1]):
                w_tmp = np.copy(self.w2)
                w_tmp[k, d] = w_tmp[k, d] + epsilon
                e_plus, _, _ = self.forward_prop(x_sample, t_sample, self.w1, w_tmp)

                w_tmp = np.copy(self.w2)
                w_tmp[k, d] = w_tmp[k, d] - epsilon
                e_minus, _, _ = self.forward_prop(x_sample, t_sample, self.w1, w_tmp)

                numerical_grad_2[k, d] = (e_plus - e_minus) / (2 * epsilon)

                # Absolute norm
        print "The absolute norm for w2 is %d" % (np.amax(np.abs(gradw1 - numerical_grad_1)),)


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
    x, test, t_train, t_test = load_data()
    hidden_neurons = 200
    lamda = 0.1
    eta = 0.2
    iter = 10
    tol = 0.01
    nn = NeuralNetwork(x, 1, hidden_neurons, 0.1, iter, t_train, eta, tol)
    #nn.train()
    #nn.test(t_test)
    nn.gradcheck()
