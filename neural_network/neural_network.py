from typing import Callable
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from neural_network.activation_functions import Sigmoid, Tanh, Relu, LeakyRelu
from backpropagation import Backpropagation


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
