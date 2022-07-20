"""Activation functions"""

import numpy as np

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