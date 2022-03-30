from typing import Callable
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def destandardise(x: np.ndarray, max_value: float, min_value: float):
    """Destandardises data using minimum and maximum values.
    
    Args:
        x: A numpy.ndarray instance of standardised data.
        max_value: A maximum value for the destandardisation formula.
        min_value: A minimum value for the destandardisation formula.
    
    Returns:
        numpy.ndarray representing destandardised values.
    """
    return ((x - 0.1) * (max_value - min_value)) / 0.8 + min_value

def mse(observed: np.ndarray, modelled: np.ndarray):
    """Calculate the Mean Squared Error.
    
    Args:
        observed: Array of the observed values.
        modelled: Array of the modelled values.
        
    Returns:
        The Mean Squared Error for the given arrays.
    """
    return np.sum(np.power(observed - modelled, 2)) / len(observed)

def rmse(observed: np.ndarray, modelled: np.ndarray):
    """Calculates the Root Mean Squared Error.
     
    Args:
        observed: Array of the observed values.
        modelled: Array of the modelled values.
        
    Returns:
        The Root Mean Squared Error for the given arrays.
    """
    return np.sqrt(np.sum(np.power(observed - modelled, 2)) / len(observed))

def msre(observed: np.ndarray, modelled: np.ndarray):
    """Calculates the Mean Squared Relative Error.
     
    Args:
        observed: Array of the observed values.
        modelled: Array of the modelled values.
        
    Returns:
        The Mean Squared Relative Error for the given arrays.
    """
    return np.sum(np.power((modelled - observed) / observed, 2)) / len(observed)

def ce(observed: np.ndarray, modelled: np.ndarray):
    """Calculates the Coefficient of Efficiency.
     
    Args:
        observed: Array of the observed values.
        modelled: Array of the modelled values.
        
    Returns:
        The Coefficient of Efficiency for the given arrays.
    """
    return 1 - (np.sum(np.power(modelled - observed, 2)) / np.sum(np.power(observed - np.mean(observed), 2)))

def rsqr(observed: np.ndarray, modelled: np.ndarray):
    """Calculates the Coefficient of Determination - .
     
    Args:
        observed: Array of the observed values.
        modelled: Array of the modelled values.
        
    Returns:
        The Mean Squared Error for the given arrays.
    """
    dividend = np.sum(np.multiply(observed - observed.mean(), modelled - modelled.mean()))
    divisor = np.sqrt(np.multiply(np.sum(np.power(observed - observed.mean(), 2)), np.sum(np.power(modelled - modelled.mean(), 2))))
    return np.power(dividend / divisor, 2)

class Sigmoid:
    """Represents Sigmoid activation function.
    """
    
    def __init__(self):
        """Initialises a sigmoid object."""
        self.vectorised_func = np.vectorize(self.func)
        self.vectorised_der = np.vectorize(self.der)
        
    def func(self, x: float):
        """Calculates output of the Sigmoid function.
        
        Args:
            x: Input to the sigmoid function.
            
        Returns:
            A float representing the output of sigmoid.
        """
        return 1 / (1 + np.e ** (-x))
    
    def der(self, x: float):
        """Calculates output of the derivative of the Sigmoid function.
        
        Args:
            x: Input to the derivative of sigmoid.
            
        Returns:
            A float representing the output of sigmoid's derivative.
        """
        return self.func(x) * (1 - self.func(x))

class Tanh:
    """Represents tanh activation function.
    """
    
    def __init__(self):
        """Initialises a tanh object."""
        self.vectorised_func = np.vectorize(self.func)
        self.vectorised_der = np.vectorize(self.der)
    
    def func(self, x: float):
        """Calculates output of the tanh function.
        
        Args:
            x: Input to the tanh function.
            
        Returns:
            A float representing the output of tanh.
        """
        return (np.e ** x - np.e ** (-x)) / (np.e ** x + np.e ** (-x))
    
    def der(self, x: float):
        """Calculates output of the derivative of the tanh function.
        
        Args:
            x: Input to the derivative of tanh.
            
        Returns:
            A float representing the output of tanh's derivative.
        """
        return 1 - self.func(x) ** 2

class Relu:
    """Represents ReLU activation function.
    """
    
    def __init__(self):
        """Initialises a Relu object."""
        self.vectorised_func = np.vectorize(self.func)
        self.vectorised_der = np.vectorize(self.der)
    
    def func(self, x: float):
        """Calculates output of the ReLU function.
        
        Args:
            x: Input to the ReLU function.
            
        Returns:
            A float representing the output of ReLU.
        """
        if x > 0:
            return x
        else:
            return 0
    
    def der(self, x: float):
        """Calculates output of the derivative of the ReLU function.
        
        Args:
            x: Input to the derivative of ReLU.
            
        Returns:
            A float representing the output of ReLU's derivative.
        """
        if x > 0:
            return 1
        else:
            return 0

class LeakyRelu:
    """Represents Leaky ReLU activation function.
    """
    
    def __init__(self):
        """Initialises a LeakyRelu object."""
        self.vectorised_func = np.vectorize(self.func)
        self.vectorised_der = np.vectorize(self.der)
    
    def func(self, x: float):
        """Calculates output of the Leaky ReLU function.
        
        Args:
            x: Input to the Leaky ReLU function.
            
        Returns:
            A float representing the output of Leaky ReLU.
        """
        if x > 0:
            return x
        else:
            return 0.01 * x
    
    def der(self, x: float):
        """Calculates output of the derivative of the Leaky ReLU function.
        
        Args:
            x: Input to the derivative of Leaky ReLU.
            
        Returns:
            A float representing the output of Leaky ReLU's derivative.
        """
        if x > 0:
            return 1
        else:
            return 0.01

class Backpropagation:
    """Backpropagation algorithm for training neural networks.
    
    Attributes:
        neural_network: NeuralNetwork instance being trained.
        epochs: Number of epochs the nerual network has gone through
            during the training.
        previous_validation_error: Previous error (MSE) on the validation set.
        previous_training_error: Previous error (MSE) on the training set.
        validation_errors: List of validation errors. Used for plotting.
        training_errors: List of training errors. Used for plotting.
        tested_at_epochs: List epochs training and validation errors were
            calculated at.
        delta: List of arrays of delta values for neurons in the network.
            Each element of the list is an array representing delta values
            for a particular layer.
    """
    
    def __init__(
        self,
        neural_network
    ):
        """Initialises Backpropagation object."""
        self.neural_network = neural_network
        self.epochs = 0
        self.previous_validation_error = np.inf
        self.previous_training_error = np.inf
        self.validation_errors = []
        self.training_errors = []
        self.tested_at_epochs = []
        self.delta = []
        # Initialise the delta attribute.
        for layer in self.neural_network.layers:
            self.delta.append(None)
    
    def train(
        self,
        training_set: np.ndarray,
        validation_set: np.ndarray,
        validation_frequency: int,
        max_value: float,
        min_value: float,
        learning_rate: float,
        epoch_limit: int,
        momentum: float = 0.0,
        bold_driver: int = np.inf,
        simulated_annealing: bool = False,
        weight_decay: bool = False
    ):
        """Trains NeuralNetwork instance using Backpropagation.
        
        Args:
            training_set: Set that instance is trained on.
            validation_set: Set the instance is tested on during training.
            validation_frequency: Frequency of testing on validation set.
                Expressed in epochs.
            max_value: A maximum value for the destandardisation formula.
                Used for testing.
            min_value: A minimum value for the destandardisation formula.
                Used for testing.
            learning_rate: Learning rate.
            epoch_limit: The maximum number of epochs to train the instace for.
            momentum: Value of the alpha coefficient for Momentum.
                If 0, momentum is not used.
            bold_driver: Frequency of adjusting learning rate using
                bold driver. If np.inf (infinity), bold driver is not used.
            simulated_annealing: Boolean flag indicating if simulated
                annealing should be used.
            weight_decay: Boolean flag indicating if weight decay should
                be used.
        """
        # Run for the specified number of epochs.
        for i in range(epoch_limit):
            if simulated_annealing:
                # Anneal the learning rate.
                learning_rate = self.simulated_annealing(
                    start_rate=learning_rate,
                    end_rate=0.01,
                    epoch_limit=epoch_limit,
                    epochs_passed=self.epochs
                )
            self.epochs = i + 1
            # Loop through the training set and perform forward pass,
            # backward pass, and weights/biases update.
            for training_example in training_set:
                # Split individual examples into inputs (item) and label (c).
                item, c = np.hsplit(training_example, [training_set.shape[1] - 1])
                item = item.reshape(1, -1)
                c = c.reshape(1, -1)
                # Perform the backpropagation's steps.
                self.forward_pass(item)
                self.backward_pass(c, self.epochs, learning_rate, weight_decay)
                self.update_weights(item, learning_rate, momentum)
                
            # Bold driver.
            if (i + 1) % bold_driver == 0:
               # Test against the training set.
                observed, predicted = self.neural_network.test(training_set, max_value, min_value)
                current_training_error = mse(observed, predicted)
                learning_rate = self.bold_driver(
                    current_training_error,
                    learning_rate
                )
                self.previous_training_error = current_training_error
                for layer in self.neural_network.layers:
                    layer.save_weights_and_biases()
            
            # Test on the training and validation sets.
            if validation_set is not None and (i + 1) % validation_frequency == 0:
                # Test on the training set.
                observed_test, predicted_test = self.neural_network.test(
                    training_set,
                    max_value,
                    min_value
                )
                current_test_error = mse(observed_test, predicted_test)
                self.training_errors.append(current_test_error)
                # Test on the validation set.
                observed, predicted = self.neural_network.test(
                    validation_set,
                    max_value,
                    min_value
                )
                current_validation_error = mse(observed, predicted)
                self.validation_errors.append(current_validation_error)
                self.tested_at_epochs.append(self.epochs)
                print(f"Current Validation Error: {current_validation_error}")
                # Terminate if the validation error has started to increase.
                if self.previous_validation_error < current_validation_error:
                    for layer in self.neural_network.layers:
                        layer.restore_weights_and_biases()
                    break
                else:
                    self.previous_validation_error = current_validation_error
                    for layer in self.neural_network.layers:
                        layer.save_weights_and_biases()
        
    
    def forward_pass(self, inputs):
        """Performs forward pass through the network.
        
        Args:
            inputs: Vector of values represneting a training example.
        """   
        for i, layer in enumerate(self.neural_network.layers):
            if i == 0:
                layer.forward_pass(inputs)
            else:
                previous_layer = self.neural_network.layers[i-1]
                layer.forward_pass(previous_layer.output)
                
        
    def backward_pass(
        self,
        c,
        epochs_passed,
        learning_rate,
        weight_decay: bool = False
    ):
        """Performs backward pass through the network.

        Args:
            c: The label for the training example.
            epochs_passed: The number of epochs passed.
            learning_rate: The learning rate.
            weight_decay: Boolean flag singaling use of weight decay.
                Weight decay is used if the flag is set to True.
        """
        reversed_delta = list(reversed(self.delta))
        reversed_layers = list(reversed(self.neural_network.layers))
        for i, layer in enumerate(reversed_layers):
            if i == 0:
                # output layer backward pass
                if weight_decay:
                    reversed_delta[i] = np.multiply(
                        c - layer.output + self.weight_decay(
                            epochs_passed,
                            learning_rate
                        ),
                        layer.activation_function.vectorised_der(layer.sum)
                    )
                else:
                    reversed_delta[i] = np.multiply(
                        c - layer.output,
                        layer.activation_function.vectorised_der(layer.sum)
                    )
            else:
                # hidden layer backward pass
                next_layer = reversed_layers[i-1]
                reversed_delta[i] = np.multiply(
                    np.dot(next_layer.weights, reversed_delta[i-1].T).T,
                    layer.activation_function.vectorised_der(layer.sum)
                )
        self.delta = list(reversed(reversed_delta))
            
    def update_weights(
        self,
        inputs: np.ndarray,
        learning_rate: float,
        momentum: float
    ):
        """Updates weights in the network.
        
        Args:
            inputs: Inputs to the network.
            learning_rate: Learning rate.
            momentum: Value of the alpha coefficient for Momentum.
                If 0, momentum is not used.
        """
        for i, layer in enumerate(self.neural_network.layers):
            if momentum > 0:
                previous_weights = layer.weights.copy()
                previous_biases = layer.biases.copy()
            if i == 0:
                layer.weights = layer.weights + learning_rate * np.dot(inputs.T, self.delta[i])
            else:
                previous_layer = self.neural_network.layers[i-1]
                layer.weights = layer.weights + learning_rate * np.dot(
                    previous_layer.output.T,
                    self.delta[i]
                )
            layer.biases = layer.biases + learning_rate * self.delta[i]
            if momentum > 0:
                weights_delta = layer.weights - previous_weights
                biases_delta = layer.biases - previous_biases
                layer.weights = layer.weights + momentum * weights_delta
                layer.biases = layer.biases + momentum * biases_delta
            
    def bold_driver(
        self,
        current_training_error: float,
        learning_rate: float
    ):
        """Implements the Bold Driver extension.
        
        Args:
            current_training_error: Current training error.
            learning_rate: The learning rate.

        Returns:
            A float representing new learning rate.
        """
        if ((current_training_error / self.previous_training_error) * 100) >= 104:
            # Decrease the learning rate if the error has increased.
            learning_rate = learning_rate * 0.7
            for layer in self.neural_network.layers:
                layer.restore_weights_and_biases()
            print("Error increased")
        elif ((current_training_error / self.previous_training_error) * 100) <= 96:
            # Increase the learning rate if the error has decreased.
            learning_rate = learning_rate * 1.05
            print("Error decreased")
        # Check if the learning rate is the range.
        if learning_rate < 0.01:
            learning_rate = 0.01
        elif learning_rate > 0.5:
            learning_rate = 0.5
        return learning_rate
                
            
    def simulated_annealing(
        self,
        start_rate: float,
        end_rate: float,
        epoch_limit: int,
        epochs_passed: int
    ) -> float:
        """Returns annealed value of the learning rate.

        Args:
            start_rate: Initial learning rate value.
            end_rate: Final learning rate value.
            epoch_limit: Limit of epochs.
            epochs_passed: Number of epochs that have elapsed.

        Returns:
            A float representing the annealed learning rate.
        """
        divisor = 1 + np.e ** (10 - (20 * epochs_passed) /  epoch_limit)
        return end_rate + (start_rate - end_rate) * (1 - (1 / divisor))
    
    def weight_decay(
        self,
        epochs_passed: int,
        learning_rate: float
    ) -> float:
        """Calculates the penalty term for the error function.
        
        Args:
            epochs_passed: Number of epochs that have elapsed.
            learning_rate: Learning rate value.

        Returns:
            The penalty term for the error function.
        """
        weights_and_biases_sum = 0
        n = 0
        for layer in self.neural_network.layers:
            weights_and_biases_sum += np.sum(
                np.power(
                    np.vstack((layer.weights, layer.biases)),
                    2
                )
            )
            n += layer.weights.shape[0] * layer.weights.shape[1] + layer.biases.shape[1]
        omega = (1 / (2 * n)) * weights_and_biases_sum
        regularisation_parameter = 1 / (learning_rate * epochs_passed)
        return regularisation_parameter * omega

np.random.seed(0)


class Layer:
    """Layer of a neural network.
    
    Attributes:
        weights: set of weights of the layer.
        biases: set of biases of the layer.
        activation_function: Activation Function of the layer.
        number_of_neurons: Number of neurons on the layer.
        output: the most recent output of the layer.
        saved_weights: Weights saved at previous validaton point.
        saved_biases: Biases saved at previous validaton point.
    """
    def __init__(self,
                 number_of_inputs: int,
                 number_of_neurons: int,
                 activation_function
                ):
        """Initialises a NeuralNetwork instance."""
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        random_generator = np.random.default_rng(5)
        low = -2 / number_of_inputs
        high = 2 / number_of_inputs
        self.weights = random_generator.uniform(
            low=low,
            high=high,
            size=(number_of_inputs, number_of_neurons)
        )
        self.saved_weights = self.weights.copy()
        self.biases = random_generator.uniform(
            low=low,
            high=high,
            size=(1, number_of_neurons)
        )
        self.saved_biases = self.biases.copy()
    
    def forward_pass(self, inputs: np.ndarray):
        """Performs the forward pass through the layer.
        
        Args:
            inputs: Inputs to the layer.
        """
        self.sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_function.vectorised_func(self.sum)
            
    def save_weights_and_biases(self):
        """Saves current weights and biases to keep them after updating."""
        self.saved_weights = self.weights.copy()
        self.saved_biases = self.biases.copy()
        
    def restore_weights_and_biases(self):
        """Restores saved weights and biases."""
        self.weights = self.saved_weights
        self.biases = self.saved_biases     


class NeuralNetwork:
    """A neural network.
    
    Attributes:
        number_of_inputs: Number of inputs to the network.
        layers: List of network's hidden layers and the output layer.
    """
    
    def __init__(
        self,
        number_of_inputs: int,
        network_architecture,
    ):
        """Initialises a NeuralNetwork instance.
        
        Args:
            number_of_inputs: The number of inputs to the network.
            network_architecture: List of lists representing layers.
                Each inner list has two elements. First element represents
                number of hidden nodes on the layer, and the second element
                represents an activation function used for the layer. The 
                last element of the outer list is considered to be the
                output layer.
        """
        self.number_of_inputs = number_of_inputs
        self.layers = []
        for i, item in enumerate(network_architecture):
            if i == 0:
                number_of_layer_inputs = number_of_inputs
            else:
                number_of_layer_inputs = self.layers[-1].number_of_neurons
            layer = Layer(
                number_of_inputs=number_of_layer_inputs,
                number_of_neurons=item[0],
                activation_function=item[1]
            )
            self.layers.append(layer)
                       
    def test(
        self,
        test_set: np.ndarray,
        max_value: float,
        min_value: float
    ) -> list:
        """Tests the neural network and returns both observed and modelled values.
        
        Args:
            test_set: Array of test examples (including labels).
            max_value: A maximum value for the destandardisation formula.
            min_value: A minimum value for the destandardisation formula.
            
        Returns:
            Two-element list with observed values being the first
                element and modelled being the second.
        """
        predicted_values = np.empty(shape=(test_set.shape[0], 1))
        correct_values = test_set[:, test_set.shape[1] - 1]
        correct_values = correct_values.reshape(-1, 1)
        for i in range(len(test_set)):
            # Split individual examples into inputs (item) and label (c).
            item, c = np.hsplit(test_set[i], [test_set.shape[1] - 1])
            item = item.reshape(1, -1)
            predicted_values[i] = self.predict(item)
        correct_values = destandardise(correct_values, max_value=max_value, min_value=min_value)
        predicted_values = destandardise(predicted_values, max_value=max_value, min_value=min_value)
        return [correct_values, predicted_values]
       
    def save_network(self, file_path: str):
        """Saves the network to the file in JSON format.
        
        Args:
            file_path: File path to the target file.
        """
        layers_details = []
        for layer in self.layers:
            layer_details = {
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "activation_function": layer.activation_function.__class__.__name__
            }
            layers_details.append(layer_details)

        neural_network = {
            "layers": layers_details
        }
        try:
            with open(file_path, "w") as f:
                json.dump(neural_network, f)
        except IOError as e:
            print(e)
        except Exception as e:
            print(f"Unexpected exception: {e}")
            
    @classmethod
    def load_network(cls, file_path: str):
        """Returns the network loaded from the file in JSON format.
        
        Args:
            file_path: File path to the target file.
        
        Returns:
            A NeuralNetwork instance.
        """
        try:
            with open(file_path, "r") as f:
                nn_details = json.load(f)
            layers = []
            for layer_details in nn_details['layers']:
                weights = np.array(layer_details['weights'])
                biases = np.array(layer_details['biases'])
                layer = Layer(
                    number_of_inputs=weights.shape[0],
                    number_of_neurons=weights.shape[1],
                    activation_function=eval(f"{layer_details['activation_function']}()")
                )
                layer.weights = weights
                layer.biases = biases
                layers.append(layer)
        except IOError as e:
            print(e)
            return None
        except Exception as e:
            print(f"Unexpected exception: {e}")
            return None
        architecture = []
        for layer in layers:
            layer_details = tuple([layer.weights.shape[1], layer.activation_function])
            architecture.append(layer_details)
        neural_network = cls(
            number_of_inputs=layers[0].weights.shape[0],
            network_architecture=architecture
        )
        neural_network.layers = layers
        return neural_network
                
    
    def predict(self, inputs: np.ndarray):
        """Predicts value for given predictor values.
        
        Args:
            inputs: Inputs to the network.
        
        Returns:
            A float representing the predicted value.
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward_pass(inputs)
            else:
                previous_layer = self.layers[i-1]
                layer.forward_pass(previous_layer.output)
        return self.layers[-1].output



# === EXAMPLE ===

if __name__ == "__main__":

    # Load min and max values for the destandardisation process.
    with open("standardisation.json", "r") as f:
        min_max_values = json.load(f)

    min_value = min_max_values["min"]
    max_value = min_max_values["max"]

    # Load data.
    training_set = pd.read_csv("data/training-set.csv")
    training_set = training_set.to_numpy() # Convert to a numpy array.
    training_set = training_set[:, 1:] # Get rid of the index column.

    validation_set = pd.read_csv("data/validation-set.csv")
    validation_set = validation_set.to_numpy() # Convert to a numpy array.
    validation_set = validation_set[:, 1:] # Get rid of the index column.

    test_set = pd.read_csv("data/test-set.csv")
    test_set = test_set.to_numpy() # Convert to a numpy array.
    test_set = test_set[:, 1:] # Get rid of the index column.

    # Create a neural network.
    neural_network = NeuralNetwork(
        number_of_inputs=training_set.shape[1]-1,
        network_architecture=[[6, Sigmoid()], [1, Sigmoid()]]
    )

    # Create a backpropagation instance.
    backpropagation = Backpropagation(
        neural_network=neural_network
    )
       

    # Train.
    start_time = time.perf_counter()
    backpropagation.train(
        training_set=training_set,
        validation_set=validation_set,
        validation_frequency=10,
        max_value=max_value,
        min_value=min_value,
        learning_rate=0.2,
        epoch_limit=500,
        momentum=0.0,
        bold_driver=np.inf,
        simulated_annealing=False,
        weight_decay=False
    )
          
    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time} seconds")
    print(f"Smallest Validation Error: {backpropagation.previous_validation_error}")
    print(f"Number of epochs: {backpropagation.epochs}")
    observed, predicted = neural_network.test(validation_set, max_value, min_value)
    print(f"{mse(observed, predicted)},")
    print(f"{rmse(observed, predicted)},")
    print(f"{msre(observed, predicted)},")
    print(f"{ce(observed, predicted)},")
    print(f"{rsqr(observed, predicted)}\n")
