"""Backpropagation training algorithm."""
import numpy as np

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
        elif ((current_training_error / self.previous_training_error) * 100) <= 96:
            # Increase the learning rate if the error has decreased.
            learning_rate = learning_rate * 1.05
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