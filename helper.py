import numpy as np
import scipy.special
import imageio


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
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

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
        self.who += self.update_weight(output_errors, final_outputs, hidden_outputs)
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

    # back query
    def back_query(self, targets_list):
        final_outputs = np.array(targets_list, ndmin=2).T

        final_inputs = self.inverse_activation_function(final_outputs)

        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to 0.99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to 0.99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


def image_rescaller(file_name):
    """
     The reason for the (255.0 - img_array.reshape(784)) is that it is conventional for 0 to mean black and 255 to mean white,
     but the MNIST data set has this the opposite way around,
     so we have to reverse the values to match what the MNIST data does. 
     
     ((img_data / 255.0 + 0.99) + 0.01) is to rescale the data values so they range from 0.01 to 1.0. 
    """
    img_array = imageio.imread(file_name, as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    return (img_data / 255.0 * 0.99) + 0.01


if __name__ == "__main__":
    pass

