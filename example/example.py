"""Example usage of the neural_network package."""
from neural_network.neural_network import *
from neural_network.backpropagation import Backpropagation
from neural_network.activation_functions import Sigmoid

if __name__ == "__main__":

    # Load min and max values for the destandardisation process.
    with open("standardisation.json", "r") as f:
        min_max_values = json.load(f)

    min_value = min_max_values["min"]
    max_value = min_max_values["max"]

    # Load data.
    training_set = pd.read_csv("training-set.csv")
    training_set = training_set.to_numpy() # Convert to a numpy array.
    training_set = training_set[:, 1:] # Get rid of the index column.

    validation_set = pd.read_csv("validation-set.csv")
    validation_set = validation_set.to_numpy() # Convert to a numpy array.
    validation_set = validation_set[:, 1:] # Get rid of the index column.

    test_set = pd.read_csv("test-set.csv")
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