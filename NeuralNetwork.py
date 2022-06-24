import numpy as np
import scipy.special


class NeuralNetwork:
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # generate weights between input and hidden layers
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)
        )

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # update weight function
        self.update_weight = lambda Ek, Ok, Oj: self.lr * np.dot(
            Ek * Ok * (1 - Ok), Oj.T
        )

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # convert targets list to 2d array
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output = target - actual
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.update_weight(output_errors, final_outputs, hidden_errors)
        # update the weights for the links between the input and hidden layers
        self.wih += self.update_weight(hidden_errors, hidden_outputs, inputs)

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == "__main__":
    pass
