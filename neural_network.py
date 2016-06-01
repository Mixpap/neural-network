from __future__ import division
import numpy as np


def softmax(w):
    max_of_rows = np.max(w, 1)
    m = np.array([max_of_rows, ] * w.shape[1]).T
    w = w - m
    w = np.exp(w)
    return w / (np.array([np.sum(w, 1), ] * w.shape[1]).T)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def load_data():
    """
    Loads the MNIST dataset. Reads the training files and creates matrices.
    :return: train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    train_truth: the matrix consisting of one hot vectors on each row(ground truth for training)
    test_truth: the matrix consisting of one hot vectors on each row(ground truth for testing)
    """
    train_files = ['data/train%d.txt' % (i,) for i in range(10)]
    test_files = ['data/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load train data in N*D array (60000x784 for MNIST) divided by 255 to achieve normalization
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print "Train data array size: ", train_data.shape
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST) divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print "Test data array size: ", test_data.shape
    tmp = []
    for i, _file in enumerate(train_files):
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    train_truth = np.array(tmp, dtype='int')
    del tmp[:]
    for i, _file in enumerate(test_files):
        with open(_file, 'r') as fp:
            for _ in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    test_truth = np.array(tmp, dtype='int')
    print "Train truth array size: ", train_truth.shape
    print "Test truth array size: ", test_truth.shape
    return train_data, test_data, train_truth, test_truth


def activation_function(case):
    """
    Returns the corresponding activation function and the derrivative of
    this function
    1. h(a) = log(1 + exp^a)
    2. h(a) = (exp^a - exp^(-a))/(exp^a + exp^(-a))
    3. h(a) = cos(a)
    :param case: The number of the activation function
    :return: the activation function to be used by the neurons
     in the hidden layer
    """
    #
    def logarithmic(a):
        """
        programmed like this for numerical stability
        :param a:
        :return:
        """
        m = np.maximum(0, a)
        return m + np.log(np.exp(-m) + np.exp(a - m))

    def grad_logarithm(a):
        return sigmoid(a)

    def tanh(a):
        return 2 * sigmoid(2 * a) - 1

    def grad_tanh(a):
        return 1 - tanh(a) ** 2


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
                 hidden_neurons, lamda, iter, t, eta, tol):
        self.iter = iter
        self.x = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
        self.t = t
        self.lamda = lamda
        self.eta = eta
        self.tol = tol
        self.hidden_neurons = hidden_neurons
        self.number_of_outputs = t.shape[1]
        self.hidden_activation, self.grad_activation = activation_function(
            hidden_layer_activation_function)
        # initialize weights
        self.w1 = np.random.rand(self.hidden_neurons, np.size(self.x, 1)) * 0.2 - 0.1
        print "W(1) is of size M x(D+1) :", self.w1.shape
        # try W2 as zeros
        self.w2 = np.random.rand(self.number_of_outputs, self.hidden_neurons + 1)
        print "W(2) is of size K x(M+1) :", self.w2.shape

    def forward_prop(self, x, t, w1, w2):
        """
        feed forward and get error, gradw2 and gradw1
        :param x:
        :param t:
        :param w1:
        :param w2:
        :return: error, gradw1 and gradw2
        """
        # feed forward
        z = self.hidden_activation(np.dot(x, w1.T))
        # add bias to z
        z_with_bias = np.concatenate((np.ones((z.shape[0], 1)), z), axis=1)
        # multiplication and transpose
        y = np.dot(z_with_bias, w2.T)
        max_error = np.max(y, 1)
        E = np.sum(t * y) - np.sum(max_error) - \
            np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * self.number_of_outputs).T), 1))) - \
            (0.5 * self.lamda) * np.sum(w2 * w2)
        s = softmax(y)
        # calculate gradient of W2
        gradw2 = np.dot((t - s).T, z_with_bias) - self.lamda * w2
        # calculate gradient of W1 (we get rid of the bias from w2)
        gradw1 = (w2[:, 1:].T.dot((t - s).T) * self.grad_activation(x.dot(w1.T)).T).dot(x)
        return E, gradw1, gradw2

    def train(self):
        e_old = -np.inf
        for i in range(iter):
            error, gradw1, gradw2 = self.forward_prop(self.x, self.t, self.w1, self.w2)
            print "iteration #", i, ",error =", error
            if np.absolute(error - e_old) < self.tol:
                break
            self.w1 += self.eta * gradw1
            self.w2 += self.eta * gradw2
            e_old = error
        return error

    def test(self, test, test_truth):
        # add bias to test
        test = np.concatenate((np.ones((test.shape[0], 1)), test), axis=1)
        x = test
        # feed forward
        z = self.hidden_activation(np.dot(x, self.w1.T))
        # add bias to z
        z_with_bias = np.concatenate((np.ones((z.shape[0], 1)), z), axis=1)
        # multiplication and transpose
        y = np.dot(z_with_bias, self.w2.T)
        s = softmax(y)
        decision = np.argmax(s, 1)
        error_count = 0
        for i in range(len(test_truth)):
            if np.argmax(test_truth[i]) != decision[i]:
                error_count += 1
        print "Error is ", error_count / test_truth.shape[0] * 100, " %"


    def gradcheck(self):
        epsilon = 1e-6
        _list = np.random.randint(self.x.shape[0], size=5)
        x_sample = np.array(self.x[_list, :])
        t_sample = np.array(self.t[_list, :])
        E, gradw1, gradw2 = self.forward_prop(x_sample, t_sample, self.w1, self.w2)
        print "gradw2 : ", gradw2.shape
        print "gradw1 : ", gradw1.shape
        numerical_grad_1 = np.zeros(gradw1.shape)
        numerical_grad_2 = np.zeros(gradw2.shape)
        # gradcheck for w1
        for k in range(0, numerical_grad_1.shape[0]):
            for d in range(0, numerical_grad_1.shape[1]):
                w_tmp = np.copy(self.w1)
                w_tmp[k, d] += epsilon
                e_plus, _, _ = self.forward_prop(x_sample, t_sample, w_tmp, self.w2)

                w_tmp = np.copy(self.w1)
                w_tmp[k, d] -= epsilon
                e_minus, _, _ = self.forward_prop(x_sample, t_sample, w_tmp, self.w2)
                numerical_grad_1[k, d] = (e_plus - e_minus) / (2 * epsilon)
        # Absolute norm
        print "The difference estimate for gradient of w1 is : ", np.amax(np.abs(gradw1 - numerical_grad_1))
        # gradcheck for w2
        for k in range(0, numerical_grad_2.shape[0]):
            for d in range(0, numerical_grad_2.shape[1]):
                w_tmp = np.copy(self.w2)
                w_tmp[k, d] += epsilon
                e_plus, _, _ = self.forward_prop(x_sample, t_sample, self.w1, w_tmp)

                w_tmp = np.copy(self.w2)
                w_tmp[k, d] -= epsilon
                e_minus, _, _ = self.forward_prop(x_sample, t_sample, self.w1, w_tmp)
                numerical_grad_2[k, d] = (e_plus - e_minus) / (2 * epsilon)
        # Absolute norm
        print "The difference estimate for gradient of w2 is : ", np.amax(np.abs(gradw2 - numerical_grad_2))


if __name__ == '__main__':
    print "Welcome to the world of Neural Networks.\nLets classify the MNIST dataset..."
    hidden_neurons = int(raw_input("How many neurons would you like: "))
    act_function = int(raw_input("Which activation function would you like to "
                                 "use?:\n1.logarithmic\n2.tanh\n3.cosine\n(insert number 1-3)\n "))
    gradcheck_answer = raw_input("Would you like to run gradient check before? (Y/N): ")
    hidden_neurons = int(hidden_neurons)
    act_function = int(act_function)
    x, test, train_truth, test_truth = load_data()
    hidden_neurons = 500
    lamda = 0.1
    eta = 0.5 / x.shape[0]
    iter = 100
    tol = 0.000001
    nn = NeuralNetwork(x, act_function, hidden_neurons, lamda, iter, train_truth, eta, tol)
    if gradcheck_answer.lower() == "y".lower():
        nn.gradcheck()
    nn.train()
    nn.test(test, test_truth)

    neurons = [100, 200, 300, 400, 500]
    for neurons in neurons:
        nn = NeuralNetwork(x, 1, neurons, lamda, iter, train_truth, eta, tol)
        cost = nn.train()
        error = nn.test(test, test_truth)
        s = "For " + str(iter) + " iterations, and " + str(neurons) + " neurons, the value of the cost function  is " + \
            str(cost) + ", and the error rate is " + str(error) + "%\n"
        with open('results.txt', 'a') as fp:
            fp.write(s)
